import math
from pathlib import Path
import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR

import wandb

from asdfghjkl import KFAC
from asdfghjkl import SHAPE_KRON
from asdfghjkl.fisher import LOSS_CROSS_ENTROPY

from common.py.ml.datasets import IMAGENET, MNIST
from common.py.ml.models import MNISTToy
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='kfac')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--dataset',
                        default='IMAGENET',
                        type=str,
                        choices=['IMAGENET', 'MNIST'])
    parser.add_argument('--data-path',
                        default='/sqfs/work/jh210024/data/ILSVRC2012',
                        type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--val-batch-size', default=2048, type=int)
    parser.add_argument('--label-smoothing', default=0.1, type=float)

    parser.add_argument('--model',
                        default='resnet50',
                        type=str,
                        choices=['resnet50', 'MNISTToy'])

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.8)
    parser.add_argument('--warmup-factor', type=float, default=0.125)
    parser.add_argument('--warmup-epochs', type=float, default=5)
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[15, 25, 30])

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.00005)
    parser.add_argument('--cov-update-freq', type=int, default=10)
    parser.add_argument('--inv-update-freq', type=int, default=100)
    parser.add_argument('--ema-decay', type=float, default=0.05)
    parser.add_argument('--damping', type=float, default=0.001)
    parser.add_argument('--kl-clip', type=float, default=0.001)

    return parser.parse_args()


def to_vector(x):
    return nn.utils.parameters_to_vector(x)


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
        opt.zero_grad(set_to_none=True)

        if args.cov_update_freq != -1 and i % args.cov_update_freq == 0:
            scale = 1. / args.world_size
            loss, outputs = kfac.accumulate_curvature(inputs,
                                                      targets,
                                                      ema_decay=args.ema_decay,
                                                      calc_emp_loss_grad=True,
                                                      scale=1. /
                                                      args.world_size)
            if args.world_size > 1:
                kfac.reduce_curvature()
                for p in model.parameters():
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= args.world_size
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        if args.inv_update_freq != -1 and i % args.inv_update_freq == 0:
            kfac.update_inv(args.damping)

        #                              kl_clip
        # kl_clip: grad *= sqrt(---------------------)
        #                        |sum(ng*grad)*lr^2|
        grad = to_vector([p.grad for p in kfac.parameters_for(SHAPE_KRON)])
        kfac.precondition()
        ng = to_vector([p.grad for p in kfac.parameters_for(SHAPE_KRON)])
        vg_sum = ((ng * grad).sum() * lr**2).item()
        nu = min(1.0, (args.kl_clip / abs(vg_sum))**0.5)
        for p in kfac.parameters_for(SHAPE_KRON):
            p.grad.data *= nu
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
        wandb.init(project='kfac')
        wandb.run.name = f'{args.dataset}/{args.model}/{args.name}'

    # ========== DATA ==========
    if args.dataset == 'IMAGENET':
        dataset = IMAGENET(args)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.dataset == 'MNIST':
        dataset = MNIST(args)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    # ========== MODEL ==========
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=dataset.num_classes)
    elif args.model == 'MNISTToy':
        model = MNISTToy()
    else:
        raise ValueError(f'Unknown model {args.model}')
    model.to(args.device)

    # ========== OPTIMIZER ==========
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    kfac = KFAC(model,
                'fisher_emp',
                loss_type=LOSS_CROSS_ENTROPY,
                ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d])
    for module in kfac.modules_for(SHAPE_KRON):
        print(f"Registered {module}")

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])

    # ========== LEARNING RATE SCHEDULER ==========
    if args.warmup_epochs > 0:
        lr_scheduler = SequentialLR(opt, [
            LinearLR(opt, args.warmup_factor, total_iters=args.warmup_epochs),
            MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1),
        ], [args.warmup_epochs])
    else:
        lr_scheduler = MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1)

    # ========== TRAINING ==========
    for e in range(args.epochs):
        train(e, dataset, model, criterion, opt, kfac, args)
        torch.cuda.empty_cache()
        test(e, dataset, model, criterion, args)
        torch.cuda.empty_cache()
        lr_scheduler.step()

    torch.save(model.state_dict(), f"{args.dir}/{args.model}")
