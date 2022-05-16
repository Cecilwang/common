from pathlib import Path
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import wandb

from common.py.ml.datasets import define_dataset_arguments, create_dataset
from common.py.ml.models import define_model_arguments, create_model
from common.py.ml.pruning import define_pruning_arguments, create_pruner
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric
from common.py.ml.util.util import to_vector, list_module


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Pruning')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--device', default='cpu', type=str)

    define_dataset_arguments(parser)
    define_model_arguments(parser)

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--lr-decay-epoch', nargs='+', type=int, default=[40])
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    define_pruning_arguments(parser)

    return parser.parse_args()


def polynomial_schedule(start, end, i, n):
    scale = end - start
    progress = min(float(i) / n, 1.0)
    remaining_progress = (1.0 - progress)**2
    return end - scale * remaining_progress


def train(epoch, dataset, model, opt, pruner, args):
    dataset.train()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.train()

    lr = opt.param_groups[0]['lr']
    metric = Metric(args.device)
    for i, (inputs, targets) in enumerate(dataset.loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        opt.zero_grad()
        outputs = model(inputs)
        loss = dataset.criterion(outputs, targets)
        loss.backward()
        opt.step()

        metric.update(inputs.shape[0], loss, outputs, targets)

        #if i % 100 == 0:
        #    print(f'Epoch {epoch} {i}/{len(dataset.loader)} Train {metric}')

    if args.distributed:
        metric.sync()
    #print(f'Epoch {epoch} Train {metric} LR: {lr}')
    wandb.log({'epoch': epoch, 'train/accuracy': metric.accuracy, 'lr': lr})


def test(epoch, dataset, model, args, prefix=''):
    dataset.eval()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.eval()

    metric = Metric(args.device)

    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(dataset.loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = dataset.criterion(outputs, targets)

            metric.update(inputs.shape[0], loss, outputs, targets)

    if args.distributed:
        metric.sync()
    print(f'Epoch {epoch} {prefix} Test {metric}')
    wandb.log({'epoch': epoch, 'test/accuracy': metric.accuracy})
    return metric.accuracy


def prune(pruner, sparsity, args):
    if args.pruner == 'obs':
        sparsity = (sparsity - pruner.sparsity) / args.n_recompute
        for _ in range(args.n_recompute):
            pruner.calc_fisher(fisher_dataset.loader, args.n_batch,
                               args.damping)
            pruner.prune(pruner.sparsity + sparsity)
    else:
        pruner.prune(sparsity)


def validate(args, dataset):
    model_path = args.model_path
    args.model_path = f"{args.dir}/model"
    model = create_model(args)
    modules = list_module(
        model,
        condition=lambda x:
        (not isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.
                            LayerNorm))) and hasattr(x, 'weight'))
    n = 0
    n_zero = 0
    for k, x in modules.items():
        n += x.weight.numel()
        n_zero += len((x.weight == 0.0).nonzero())
        if x.bias is not None:
            n += x.bias.numel()
            n_zero += len((x.bias == 0.0).nonzero())
    acc = test(args.epochs, dataset, model, args, f'Val      {n_zero/n:.2f}')
    args.model_path = model_path


if __name__ == '__main__':
    args = parse_args()
    args.name = f'{args.dataset}/{args.model}/{args.pruner}/{args.sparsity}/{args.name}'
    args.dir = f'{args.dir}/{args.name}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    print(args)

    if args.rank == 0:
        wandb.init(project='pruning')
        wandb.run.name = f'{args.name}'

    dataset = create_dataset(args)
    model = create_model(args)
    pruner = create_pruner(
        args, model,
        (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm))
    if args.pruner == 'obs':
        fisher_dataset = create_dataset(args,
                                        batch_size=args.fisher_batch_size)
        fisher_dataset.train()
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1)

    sparsity = 0.0
    pretrained_acc = test(0, dataset, model, args, 'Pretrained   ')
    wandb.log({'best_acc': pretrained_acc, 'sparsity': sparsity})

    sparsity = 0.05
    prune(pruner, sparsity, args)
    best_acc = test(0, dataset, model, args, f'Pruning  {pruner.sparsity:.2f}')
    best_model_state = pruner.dump()
    wandb.log({'best_acc': best_acc, 'sparsity': sparsity})

    for e in range(args.epochs):
        if e in args.pruning_epochs:
            wandb.log({'best_acc': best_acc, 'sparsity': sparsity})
            sparsity = polynomial_schedule(0.05, args.sparsity,
                                           args.pruning_epochs.index(e) + 1,
                                           len(args.pruning_epochs))
            prune(pruner, sparsity, args)
            best_acc = test(e, dataset, model, args,
                            f'Pruning  {pruner.sparsity:.2f}')
            best_model_state = pruner.dump()
        train(e, dataset, model, opt, pruner, args)
        acc = test(e, dataset, model, args, f'Training {pruner.sparsity:.2f}')
        if acc > best_acc:
            best_acc = acc
            best_model_state = pruner.dump()
        lr_scheduler.step()
    wandb.log({
        'best_acc': best_acc,
        'sparsity': sparsity,
        'pruned_acc': best_acc,
        'drop_rate': (best_acc - pretrained_acc) / pretrained_acc
    })

    torch.save(best_model_state, f"{args.dir}/model")
    validate(args, dataset)
