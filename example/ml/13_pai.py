from pathlib import Path
import os
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import WeightedRandomSampler
import torchvision
import wandb as WANDB

from common.py.ml.datasets import define_dataset_arguments, create_dataset
from common.py.ml.models import define_model_arguments, create_model
from common.py.ml.pruning import define_pruning_arguments, create_pruner
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric
from common.py.ml.util.util import to_vector, list_module
from common.py.util.io import load

SPARSITY = []
ACC = []
wandb = None


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def n_nonzero_params(model):
    return sum(
        torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='PaI')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--rewind',
                        default='/home/sxwang/pai/rewind/0',
                        type=str)
    parser.add_argument("--pretrain",
                        dest="pretrain",
                        action="store_true",
                        default=False)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--pretrain_lr", type=float, default=None)

    define_dataset_arguments(parser)
    define_model_arguments(parser)

    # Training
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--lr', type=float, default=0.03)  #bs=128
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[60, 75])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)

    parser.add_argument('--pruning_fraction', type=float, default=0.2)
    parser.add_argument('--pruning_iteration', type=int, default=20)

    define_pruning_arguments(parser)
    return parser.parse_args()


def train(level, epoch, dataset, model, opt, lr_scheduler, args):
    dataset.train()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.train()

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
        lr_scheduler.step()

    if args.distributed:
        metric.sync()
    lr = opt.param_groups[0]['lr']
    print(f'{level} {epoch} Train A {metric.accuracy} L {metric.loss} LR {lr}')
    if level != -1:
        wandb.log(
            {
                'train/accuracy': metric.accuracy,
                'train/loss': metric.loss,
                'LR': lr
            },
            step=epoch)


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
    print(f'{level} {epoch} Test  Acc: {metric.accuracy}')
    if level != -1:
        wandb.log({
            'test/accuracy': metric.accuracy,
            'test/loss': metric.loss
        },
                  step=epoch)
    return metric.accuracy


def create_lr_scheduler(opt, lr_decay_epoch, iters_per_epoch):
    milestones = [x * iters_per_epoch for x in args.lr_decay_epoch]
    warmup_iters = milestones[0]
    milestones = [x - warmup_iters + 1 for x in milestones]
    return SequentialLR(opt, [
        LinearLR(opt, 1e-5, total_iters=warmup_iters),
        MultiStepLR(opt, milestones, gamma=0.1)
    ], [warmup_iters])


def verify(level, args, dataset):
    model = create_model(args, model_path=f'{args.dir}/{level}/model')
    ACC.append(test(-1, args.epochs, dataset, model, args))
    SPARSITY.append(1. - (n_nonzero_params(model) / n_params(model)))


def save(obj, path):
    Path('/'.join(path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def train_and_test(level, args):
    dataset = create_dataset(args)
    model = create_model(args, model_path=args.rewind)
    # Ignore Last Fully Connected and Residual Layer
    ignore = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)
    pruner = create_pruner(args, model, ignore)
    if level > 0:
        pruner.mask = torch.load(f'{args.dir}/{level}/mask')['mask']
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(opt, args.lr_decay_epoch,
                                       len(dataset.train_loader))
    for e in range(args.epochs):
        train(level, e, dataset, model, opt, lr_scheduler, args)
        test(level, e, dataset, model, args)
    save(pruner.dump(), f'{args.dir}/{level}/model')
    verify(level, args, dataset)
    return pruner


def prune(level, pruner, args):
    sparsity = (1 - pruner.sparsity) * args.pruning_fraction
    if args.pruner == 'obs':
        dataset = create_dataset(args, batch_size=args.fisher_batch_size)
        dataset.train()
        sparsity /= args.n_recompute
        for _ in range(args.n_recompute):
            pruner.calc_fisher(dataset.loader, args.n_batch, args.damping)
            pruner.prune(pruner.sparsity + sparsity)
    else:
        pruner.prune(pruner.sparsity + sparsity)
    save({'mask': pruner.mask.clone()}, f'{args.dir}/{level}/mask')


def init_model(args):
    dataset = create_dataset(args)
    model = create_model(args)
    save(model.state_dict(), f'{args.rewind}/{args.initializer}/0')
    if args.pretrain_lr is None:
        args.pretrain_lr = args.lr
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.pretrain_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(opt, args.lr_decay_epoch,
                                       len(dataset.train_loader))
    for e in range(args.pretrain_epochs):
        train(-1, e, dataset, model, opt, lr_scheduler, args)
        test(-1, e, dataset, model, args)
        save(model.state_dict(), f'{args.rewind}/{args.initializer}/{e+1}')


def new_wandb_run(args, level):
    args.level = level
    if args.rank == 0:
        global wandb
        if wandb is not None:
            wandb.finish()
        wandb = WANDB.init(project='pai',
                           name=f'{args.name}/{level}',
                           config=args)


if __name__ == '__main__':
    args = parse_args()
    args.name = f'{args.dataset}/{args.model}/{args.pruner}/{args.name}/{time.time()}'
    args.dir = f'{args.dir}/{args.name}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    print(args)

    if args.pretrain:
        init_model(args)
    else:
        for level in range(args.pruning_iteration):
            new_wandb_run(args, level)
            pruner = train_and_test(level, args)
            prune(level + 1, pruner, args)

        if args.rank == 0:
            new_wandb_run(args, 'summary')
            for s, a in zip(SPARSITY, ACC):
                wandb.log({'sparsity': s, 'accuracy': a})
