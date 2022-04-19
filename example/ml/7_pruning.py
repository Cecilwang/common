from pathlib import Path
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import wandb

#from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG

from common.py.ml.datasets import IMAGENET, MNIST
from common.py.ml.models import MNISTToy
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric
from common.py.ml.pruning import Magnitude
from common.py.ml.pruning import Movement


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Pruning')
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
    parser.add_argument('--model-path', type=str)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--lr-decay-epoch', nargs='+', type=int, default=[40])
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # Pruning
    parser.add_argument('--sparsity', type=float, default=0.95)
    parser.add_argument('--pruning_epochs',
                        nargs='+',
                        type=int,
                        default=[0, 5, 10, 15, 20, 25, 30, 40])
    parser.add_argument('--check', dest='check', action='store_true')

    subparsers = parser.add_subparsers()
    parser_magnitude = subparsers.add_parser('magnitude', help='magnitude')
    parser_magnitude.set_defaults(pruner='magnitude')

    parser_movement = subparsers.add_parser('movement', help='movement')
    parser_movement.set_defaults(pruner='movement')

    parser_obs = subparsers.add_parser('obs', help='optimal brain surgeon')
    parser_obs.set_defaults(pruner='obs')
    parser_obs.add_argument(
        '--fisher_type',
        type=str,
        default='fisher_emp',
        choices=['fisher_exact', 'fisher_mc', 'fisher_emp'])
    parser_obs.add_argument('--fisher_shape',
                            type=str,
                            default='block_wood',
                            choices=[
                                'full', 'layer_wise', 'kron', 'unit_wise',
                                'full_wood', 'block_wood'
                            ])
    parser_obs.add_argument('--damping', type=float, default=1e-4)
    parser_obs.add_argument('--n_samples', type=int, default=64)
    parser_obs.add_argument('--n_recompute', type=int, default=16)
    parser_obs.add_argument('--fisher_block_size', type=int, default=128)
    parser_obs.add_argument('--fisher_block_batch', type=int, default=10000)
    parser_obs.add_argument(
        '--kfac_fast_inv',
        dest='kfac_fast_inv',
        action='store_true',
    )
    parser_obs.add_argument(
        '--layer_normalize',
        dest='layer_normalize',
        action='store_true',
    )

    return parser.parse_args()


def polynomial_schedule(start, end, i, n):
    scale = end - start
    progress = min(float(i) / n, 1.0)
    remaining_progress = (1.0 - progress)**2
    return end - scale * remaining_progress


def train(epoch, dataset, model, criterion, opt, pruner, args):
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
        loss = criterion(outputs, targets)
        loss.backward()
        opt.step()
        pruner.apply_mask()

        metric.update(inputs.shape[0], loss, outputs, targets)

        if i % 100 == 0:
            print(f'Epoch {epoch} {i}/{len(dataset.loader)} Train {metric}')

    if args.distributed:
        metric.sync()
    print(f'Epoch {epoch} Train {metric} LR: {lr}')
    wandb.log({'epoch': epoch, 'train/accuracy': metric.accuracy, 'lr': lr})


def test(epoch, dataset, model, criterion, args, prefix=''):
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
    print(f'Epoch {epoch} {prefix} Test {metric}')
    wandb.log({'epoch': epoch, 'test/accuracy': metric.accuracy})
    return metric.accuracy


if __name__ == '__main__':
    args = parse_args()
    args.dir = f'{args.dir}/{args.dataset}/{args.model}/{args.name}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    print(args)

    if args.rank == 0:
        wandb.init(project='pruning')
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
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])

    # ========== PRUNING ==========
    if args.pruner == 'magnitude':
        pruner = Magnitude(
            model,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm))
    elif args.pruner == 'movement':
        pruner = Movement(
            model,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm))

    # ========== OPTIMIZER ==========
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1)

    sparsity = 0
    best_acc = test(0, dataset, model, criterion, args, 'Pretrained')
    pretrained_acc = best_acc
    for e in range(args.epochs + 1):
        if e in args.pruning_epochs:
            wandb.log({'best_acc': best_acc, 'sparsity': sparsity})
            best_acc = 0
            sparsity = polynomial_schedule(0.05, args.sparsity,
                                           args.pruning_epochs.index(e) + 1,
                                           len(args.pruning_epochs))
            pruner.prune(sparsity)
            test(e, dataset, model, criterion, args, f'{pruner.sparsity}')
        if e > 0:
            train(e, dataset, model, criterion, opt, pruner, args)
            best_acc = max(
                best_acc,
                test(e, dataset, model, criterion, args, f'{pruner.sparsity}'))
            lr_scheduler.step()
    wandb.log({
        'best_acc': best_acc,
        'sparsity': sparsity,
        'pruned_acc': best_acc,
        'drop_rate': (best_acc - pretrained_acc) / pretrained_acc
    })
