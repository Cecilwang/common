from pathlib import Path
import os
import time

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
from common.py.util.io import load


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def n_nonzero_params(model):
    return sum(
        torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='thesis')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--device', default='cpu', type=str)

    define_dataset_arguments(parser)
    define_model_arguments(parser)
    define_pruning_arguments(parser)

    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[60, 80, 100])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--pruning_fraction', type=float, default=0.2)
    parser.add_argument('--pruning_iteration', type=int, default=20)

    subparsers = parser.add_subparsers()

    pretrain = subparsers.add_parser('pretrain')
    pretrain.set_defaults(task='pretrain')
    pretrain.add_argument('--lr', type=float, default=0.1)  #bs=128
    parser.add_argument('--ckpt', nargs='+', type=int, default=[3])

    pat = subparsers.add_parser('pat')
    pat.set_defaults(task='pat')
    pat.add_argument('--lr', type=float, default=0.1)  #bs=128

    pai = subparsers.add_parser('pai')
    pai.set_defaults(task='pai')
    pai.add_argument('--lr', type=float, default=0.03)  #bs=128
    pai.add_argument('--rewind', default=None, type=str)

    return parser.parse_args()


def train(epoch, dataset, model, opt, lr_scheduler, args):
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
    print(f'{epoch} Train A {metric.accuracy} L {metric.loss} LR {lr}')
    wandb.log(
        {
            'train/accuracy': metric.accuracy,
            'train/loss': metric.loss,
            'LR': lr
        },
        step=epoch)


def test(epoch, dataset, model, args):
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
    print(f'{epoch} Test  Acc: {metric.accuracy}')
    wandb.log({
        'test/accuracy': metric.accuracy,
        'test/loss': metric.loss
    },
              step=epoch)
    return metric.accuracy


def create_lr_scheduler(opt, lr_decay_epoch, warmup_epochs, iters_per_epoch):
    milestones = [x * iters_per_epoch for x in args.lr_decay_epoch]
    warmup_iters = warmup_epochs * iters_per_epoch
    milestones = [x - warmup_iters + 1 for x in milestones]
    return SequentialLR(opt, [
        LinearLR(opt, 1e-5, total_iters=warmup_iters),
        MultiStepLR(opt, milestones, gamma=0.1)
    ], [warmup_iters])


def save(obj, path):
    Path('/'.join(path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def pretrain(args):
    dataset = create_dataset(args)
    model = create_model(args)
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(opt, args.lr_decay_epoch,
                                       args.warmup_epochs,
                                       len(dataset.train_loader))
    for e in range(args.epochs):
        train(-1, e, dataset, model, opt, lr_scheduler, args)
        test(-1, e, dataset, model, args)
        if e in args.ckpt:
            save(model.state_dict(), f'{args.dir}/{e+1}')
    save(model.state_dict(), f'{args.dir}/final')


if __name__ == '__main__':
    args = parse_args()
    args.dir = f'{args.dir}/{args.model}/{args.initializer}/{args.task}'
    if args.task != 'pretrain':
        args.dir = f'{args.dir}/{args.pruner}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    print(args)
    wandb.init(project='thesis', name=f'{args.dir}', config=args)

    if args.task == 'pretrian':
        pretrain(args)
    if args.task == 'pat':
        pat(args)
    if args.task == 'pai':
        pai(args)
