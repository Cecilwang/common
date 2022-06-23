import math
from pathlib import Path
import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR

import wandb

from common.py.ml.datasets import IMAGENET, MNIST, CIFAR10
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--dataset',
                        default='CIFAR10',
                        type=str,
                        choices=['IMAGENET', 'MNIST', 'CIFAR10'])
    parser.add_argument('--data-path', default='.data', type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--val-batch-size', default=2048, type=int)

    parser.add_argument('--model',
                        default='resnet18',
                        type=str,
                        choices=['resnet50', 'resnet18'])

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[60, 120, 160])

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)

    return parser.parse_args()


def train(epoch, dataset, model, criterion, opt, kfac, args):
    dataset.train()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.train()

    lr = opt.param_groups[0]['lr']
    metric = Metric(args.device)
    for i, (inputs, targets) in enumerate(dataset.loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        opt.step()

        metric.update(inputs.shape[0], loss, outputs, targets)

        if i % 100 == 0:
            print(f'Epoch {epoch} {i}/{len(dataset.loader)} Train {metric}')

    if args.distributed:
        metric.sync()
    print(f'Epoch {epoch} Train {metric} LR: {lr}')
    wandb.log(
        {
            'train/loss': metric.loss,
            'train/accuracy': metric.accuracy,
            'train/lr': lr
        }, epoch)


def test(epoch, dataset, model, criterion, args):
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
            loss = criterion(outputs, targets)

            metric.update(inputs.shape[0], loss, outputs, targets)

    if args.distributed:
        metric.sync()
    print(f'Epoch {epoch} Test {metric}')
    wandb.log({
        'test/loss': metric.loss,
        'test/accuracy': metric.accuracy
    }, epoch)


if __name__ == '__main__':
    args = parse_args()
    args.dir = f'{args.dir}/{args.dataset}/{args.model}/{args.name}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    print(args)

    if args.rank == 0:
        wandb.init(project='cifar10')
        wandb.run.name = f'{args.dataset}/{args.model}/{args.name}'

    # ========== DATA ==========
    if args.dataset == 'IMAGENET':
        dataset = IMAGENET(args)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.dataset == 'MNIST':
        dataset = MNIST(args)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.dataset == 'CIFAR10':
        dataset = CIFAR10(args)
        #criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    # ========== MODEL ==========
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=dataset.num_classes)
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(num_classes=dataset.num_classes)
    else:
        raise ValueError(f'Unknown model {args.model}')
    model.to(args.device)

    # ========== OPTIMIZER ==========
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])

    # ========== LEARNING RATE SCHEDULER ==========
    lr_scheduler = MultiStepLR(opt, args.lr_decay_epoch, gamma=0.2)

    # ========== TRAINING ==========
    for e in range(args.epochs):
        train(e, dataset, model, criterion, opt, kfac, args)
        test(e, dataset, model, criterion, args)
        lr_scheduler.step()

    torch.save(model.state_dict(), f"{args.dir}/{args.model}")
