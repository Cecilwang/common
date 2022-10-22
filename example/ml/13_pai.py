from pathlib import Path
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import WeightedRandomSampler
import torchvision
import wandb

from common.py.ml.datasets import define_dataset_arguments, create_dataset
from common.py.ml.models import define_model_arguments, create_model
from common.py.ml.pruning import define_pruning_arguments, create_pruner
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric
from common.py.ml.util.util import to_vector, list_module
from common.py.util.io import load


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='PaI')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--device', default='cpu', type=str)

    define_dataset_arguments(parser)
    define_model_arguments(parser)

    # Training
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[80, 120])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)

    parser.add_argument('--pruning_fraction', type=float, default=0.2)
    parser.add_argument('--pruning_iteration', type=int, default=20)

    return parser.parse_args()


def train(level, epoch, dataset, model, opt, args):
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

    if args.distributed:
        metric.sync()
    wandb.log({'{level}/train/accuracy': metric.accuracy}, step=epoch)


def test(level, epoch, dataset, model, args):
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
    wandb.log({'{level}/test/accuracy': metric.accuracy}, step=epoch)
    return metric.accuracy


def prune(pruner, fisher_dataset, args):
    current_sparsity = pruner.sparsity
    sparsity = (1 - current_sparsity) * args.pruning_fraction
    if args.pruner == 'obs':
        sparsity /= args.n_recompute
        for _ in range(args.n_recompute):
            pruner.calc_fisher(fisher_dataset.loader, args.n_batch,
                               args.damping)
            pruner.prune(pruner.sparsity + sparsity)
    else:
        pruner.prune(pruner.sparsity + sparsity)


if __name__ == '__main__':
    args = parse_args()
    args.name = f'{args.dataset}/{args.model}/{args.pruner}/{args.name}'
    args.dir = f'{args.dir}/{args.name}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    print(args)

    if args.rank == 0:
        wandb.init(project='pai')
        wandb.run.name = f'{args.name}'
        wandb.config.update(args)

    dataset = create_dataset(args)
    model = create_model(args)
    pruner = create_pruner(
        args, model,
        (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm))
    if args.pruner == 'obs':
        fisher_dataset = create_dataset(args,
                                        batch_size=args.fisher_batch_size)
        fisher_dataset.train()
    else:
        fisher_dataset = None

    for level in range(args.pruning_iteration):
        if level > 0:
            prune(pruner, fisher_dataset, args)

        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        lr_scheduler = MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1)
        best_acc = 0.0
        for e in range(args.epochs):
            train(level, e, dataset, model, opt, args)
            acc = test(level, e, dataset, model, args)
            best_acc = max(best_acc, acc)
            lr_scheduler.step()

        wandb.log(
            {
                'sparsity': pruner.sparsity,
                'accuracy': best_acc,
            },
            step=level,
        )
        #torch.save(model, f"{args.dir}/{level}/model")
