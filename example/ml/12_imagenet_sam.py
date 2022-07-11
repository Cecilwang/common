from datetime import datetime
import math
from pathlib import Path
import os

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
from torchvision import datasets
from torchvision import transforms

import wandb

from vit_pytorch import ViT

from sam import SAM

from common.py.ml.datasets.datasets import Dataset
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='imagenet-sam')
    parser.add_argument('--dir', default='.log', type=str)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--dataset',
                        default='IMAGENET',
                        type=str,
                        choices=['IMAGENET', 'CIFAR10'])
    parser.add_argument('--data-path', default='.data', type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--val-batch-size', default=4096, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--label-smoothing', default=0.0, type=float)

    parser.add_argument('--model',
                        default='vits16',
                        type=str,
                        choices=['vits16', 'resnet50'])
    parser.add_argument('--model-path', default=None, type=str)
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--opt',
                        type=str,
                        default='adamw',
                        choices=['sgd', 'adamw'])
    parser.add_argument('--sam', dest='sam', action='store_true')
    parser.add_argument('--no-sam', dest='sam', action='store_false')
    parser.set_defaults(sam=False)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--warmup-steps', type=float, default=10000)
    parser.add_argument('--lr-sche', type=str, default='cos', choices=['cos'])

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    parser.add_argument('--clip', type=float, default=None)

    return parser.parse_args()


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module,
                                                      "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class IMAGENET(Dataset):
    def __init__(self, args):
        self.num_classes = 1000
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.img_size = 224
        self.train_dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'train'),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]))
        self.val_dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'val'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]))
        super().__init__(args)


def train(epoch, dataset, model, opt, args):
    dataset.train()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.train()

    lr = opt.param_groups[0]['lr']
    metric = Metric(args.device)
    for i, (inputs, targets) in enumerate(dataset.loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # first forward-backward step
        enable_running_stats(model)  # <- this is the important line
        outputs = model(inputs)
        loss = dataset.criterion(outputs, targets)
        if args.distributed and args.sam:
            with model.no_sync():  # <- this is the important line
                loss.backward()
        else:
            loss.backward()

        if not args.sam:
            opt.step()
            opt.zero_grad()
        else:
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)  # <- this is the important line
            dataset.criterion(model(inputs), targets).backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.second_step(zero_grad=True)

        lr_scheduler.step()

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
    args.dir = f'{args.dir}/{args.name}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)

    if args.rank == 0:
        wandb.init(project='imagenet-sam')
        wandb.config.update(args)

    print(args)

    # ========== DATASET ==========
    if args.dataset == 'IMAGENET':
        dataset = IMAGENET(args)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
    args.num_classes = dataset.num_classes

    # ========== MODEL ==========
    if args.model == 'vits16':
        model = ViT(image_size=dataset.img_size,
                        patch_size=16,
                        num_classes=dataset.num_classes,
                        dim=384,
                        depth=12,
                        heads=6,
                        mlp_dim=384,
                        dropout=args.dropout)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=args.num_classes)
    else:
        raise ValueError(f'Unknown model {args.model}')
    if args.model_path is not None:
        model.load_state_dict(
            torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])

    # ========== OPTIMIZER ==========
    if args.opt == 'sgd':
        if not args.sam:
            opt = torch.optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            base_opt = opt
        else:
            opt = SAM(model.parameters(),
                      torch.optim.SGD,
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
            base_opt = opt.base_optimizer
    elif args.opt == 'adamw':
        if not args.sam:
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
            base_opt = opt
        else:
            opt = SAM(model.parameters(),
                      torch.optim.AdamW,
                      lr=args.lr,
                      weight_decay=args.weight_decay)
            base_opt = opt.base_optimizer
    else:
        raise ValueError(f'Unknown optimizer {args.opt}')

    # ========== LEARNING RATE SCHEDULER ==========
    if args.lr_sche == 'cos':
        pct_warm = args.warmup_steps / len(dataset.train_loader) / args.epochs
        lr_scheduler = OneCycleLR(base_opt,
                                  max_lr=args.lr,
                                  anneal_strategy=args.lr_sche,
                                  steps_per_epoch=len(dataset.train_loader),
                                  epochs=args.epochs,
                                  pct_start=pct_warm)
    else:
        raise ValueError(f'Unknown learning rate scheduler {args.lr_sche}')

    # ========== TRAINING ==========
    for e in range(args.epochs):
        train(e, dataset, model, opt, args)
        test(e, dataset, model, args)

    if args.rank == 0:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        print(f'saving to {args.dir}/{timestamp}')
        torch.save(model.state_dict(), f"{args.dir}/{timestamp}")
