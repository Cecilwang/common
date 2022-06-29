import math
from pathlib import Path
import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from asdfghjkl import KFAC
from asdfghjkl import SHAPE_KRON
from asdfghjkl.fisher import LOSS_CROSS_ENTROPY

from common.py.ml.datasets import define_dataset_arguments, create_dataset
from common.py.ml.models import define_model_arguments, create_model
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='survey')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    define_dataset_arguments(parser)
    define_model_arguments(parser)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--opt',
                        type=str,
                        default='sgd',
                        choices=['sgd', 'adam', 'adamw', 'kfac'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-sche',
                        type=str,
                        default='step',
                        choices=['cos', 'step'])
    parser.add_argument('--lr-decay-rate', type=float, default=0.2)
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[60, 120, 160])

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--cov-update-freq', type=int, default=10)
    parser.add_argument('--inv-update-freq', type=int, default=100)
    parser.add_argument('--ema-decay', type=float, default=0.05)
    parser.add_argument('--damping', type=float, default=0.001)

    return parser.parse_args()


def train(epoch, dataset, model, opt, kfac, args):
    dataset.train()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.train()

    lr = opt.param_groups[0]['lr']
    metric = Metric(args.device)
    for i, (inputs, targets) in enumerate(dataset.loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        opt.zero_grad(set_to_none=True)

        if kfac is not None and i % args.cov_update_freq == 0:
            loss, outputs = kfac.accumulate_curvature(inputs,
                                                      targets,
                                                      ema_decay=args.ema_decay,
                                                      calc_emp_loss_grad=True)
        else:
            outputs = model(inputs)
            loss = dataset.criterion(outputs, targets)
            loss.backward()

        if kfac is not None and i % args.inv_update_freq == 0:
            kfac.update_inv(args.damping)

        kfac.precondition()
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
        wandb.init(project='survey')
        wandb.run.name = f'{args.dataset}/{args.model}/{args.opt}/{args.lr_sche}/{args.lr}/{args.name}'

    dataset = create_dataset(args)
    model = create_model(args)

    # ========== OPTIMIZER ==========
    kfac = None
    if args.opt == 'sgd':
        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              nesterov=True,
                              weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        opt = torch.optim.AdamW(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    elif args.opt == 'kfac':
        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              nesterov=True,
                              weight_decay=args.weight_decay)
        kfac = KFAC(model,
                    'fisher_emp',
                    loss_type=LOSS_CROSS_ENTROPY,
                    ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d])
    else:
        raise ValueError(f'Unknown optimizer {args.opt}')

    # ========== LEARNING RATE SCHEDULER ==========
    if args.lr_sche == 'step':
        lr_scheduler = MultiStepLR(opt,
                                   args.lr_decay_epoch,
                                   gamma=args.lr_decay_rate)
    elif args.lr_sche == 'cos':
        lr_scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        raise ValueError(f'Unknown learning rate scheduler {args.lr_sche}')

    # ========== TRAINING ==========
    for e in range(args.epochs):
        train(e, dataset, model, opt, kfac, args)
        test(e, dataset, model, args)
        lr_scheduler.step()

    torch.save(model.state_dict(), f"{args.dir}/{args.model}")
