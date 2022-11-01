from datetime import datetime
import math
from pathlib import Path
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
from torchvision import datasets
from torchvision import transforms

import wandb

from vit_pytorch import ViT

from sam import SAM

from ivon import IVON

from common.py.ml.datasets import define_dataset_arguments, create_dataset
from common.py.ml.models import define_model_arguments, create_model
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='imagenet-sam')
    parser.add_argument('--dir', default='.log', type=str)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    define_dataset_arguments(parser)
    define_model_arguments(parser)
    parser.add_argument('--da-factor', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--opt',
                        type=str,
                        default='adamw',
                        choices=['sgd', 'adamw', 'ivon'])
    parser.add_argument('--sam', dest='sam', action='store_true')
    parser.add_argument('--no-sam', dest='sam', action='store_false')
    parser.add_argument('--rho', type=float, default=0.05)
    parser.set_defaults(sam=False)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--warmup_steps', type=float, default=10000)
    parser.add_argument('--cooldown_epochs', type=int, default=0)
    parser.add_argument('--lr_sche',
                        type=str,
                        default='cos',
                        choices=['cos', 'step'])
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[60, 75])

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--clip', type=float, default=None)

    parser.add_argument('--mc_samples', default=1, type=int)
    parser.add_argument('--test_mc_samples', default=16, type=int)
    parser.add_argument('--momentum_grad', default=0.9, type=float)
    parser.add_argument('--momentum_hess', type=float)
    parser.add_argument('--prior_prec', default=2.0, type=float)
    parser.add_argument('--init_dp', default=10, type=float)
    parser.add_argument('--dp', default=0.001, type=float)
    parser.add_argument('--dp_warmup_epochs', default=20, type=int)
    #parser.add_argument('--init_temp', default=0.01, type=float)
    #parser.add_argument('--temp_warmup_epochs', default=10, type=int)
    parser.add_argument('--hess_init', type=float)

    return parser.parse_args()


def cos_sche(minv, maxv, i, n, s):
    return minv + (maxv - minv) / 2 * (1 + math.cos(math.pi * (i - s) /
                                                    (n - s)))


class CosSche(object):

    def __init__(self, minv, maxv, n, warmup):
        self.minv = minv
        self.maxv = maxv
        self.n = n
        self.warmup = warmup
        self.i = -1

    def step(self):
        self.i += 1
        if self.i >= self.n:
            return self.minv
        return cos_sche(self.minv, self.maxv, self.i, self.n, self.warmup)


class StepSche(object):

    def __init__(self, initv, steps, ratio=0.1):
        self.initv = initv
        self.steps = steps
        self.ratio = ratio
        self.i = -1

    def step(self):
        self.i += 1
        if self.i in self.steps:
            self.initv *= self.ratio
        return self.initv


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

        if args.opt == 'ivon':

            def closure():
                opt.zero_grad()
                outputs = model(inputs)
                loss = dataset.criterion(outputs, targets)
                if args.distributed:
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()
                return loss, outputs

            loss, outputs = opt.step(closure)
        elif args.sam:
            enable_running_stats(model)
            outputs = model(inputs)
            loss = dataset.criterion(outputs, targets)
            if args.distributed:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.first_step(zero_grad=True)

            disable_running_stats(model)
            dataset.criterion(model(inputs), targets).backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.second_step(zero_grad=True)
        else:
            outputs = model(inputs)
            loss = dataset.criterion(outputs, targets)
            loss.backward()
            opt.step()
            opt.zero_grad()

        if epoch < args.epochs - args.cooldown_epochs:
            lr_scheduler.step()
        new_dp = dp_scheduler.step()
        for group in opt.param_groups:
            group['dampening'] = new_dp

        metric.update(inputs.shape[0], loss, outputs, targets)

    if args.distributed:
        metric.sync()
    print(f'Epoch {epoch} Train {metric} LR: {lr}')
    metric = {
        'train/loss': metric.loss,
        'train/accuracy': metric.accuracy,
        'train/lr': lr
    }
    if args.opt == 'ivon':
        metric['train/dampening'] = opt.param_groups[0]['dampening']
        i = 0
        for group in opt.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p_avg = p.data
                    m_hess = opt.state[p]['momentum_hess_buffer']
                    noises = torch.vstack(opt.state[p]['noises'])
                    noise_norms = noises.norm(dim=1)
                    metric[f'model/{i}/p_avg'] = wandb.Histogram(p_avg.cpu())
                    metric[f'model/{i}/m_hess'] = wandb.Histogram(m_hess.cpu())
                    metric[f'model/{i}/p_avg_norm'] = p_avg.norm()
                    metric[f'model/{i}/m_hess_norm'] = m_hess.norm()
                    metric[f'model/{i}/noise_min'] = wandb.Histogram(
                        noises.min(dim=0).values.cpu())
                    metric[f'model/{i}/noise_max'] = wandb.Histogram(
                        noises.max(dim=0).values.cpu())
                    metric[f'model/{i}/noise_avg'] = wandb.Histogram(
                        noises.mean(dim=0).cpu())
                    metric[f'model/{i}/noise_norm_min'] = noise_norms.min()
                    metric[f'model/{i}/noise_norm_max'] = noise_norms.max()
                    metric[f'model/{i}/noise_norm_avg'] = noise_norms.mean()
                    opt.state[p]['noises'] = []
                    i += 1
    wandb.log(metric, epoch)


def test(epoch, dataset, model, args):
    dataset.eval()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.eval()

    metric = Metric(args.device)
    mc_metric = Metric(args.device)

    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(dataset.loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = dataset.criterion(outputs, targets)

            metric.update(inputs.shape[0], loss, outputs, targets)

            if args.opt == 'ivon' and args.test_mc_samples > 0:
                sampled_probs = []
                sampled_losses = []

                for i in range(args.test_mc_samples):
                    with opt.sampled_params():
                        sampled_logits = model(inputs)
                        sampled_losses.append(
                            dataset.criterion(sampled_logits, targets))
                        sampled_probs.append(F.softmax(sampled_logits, dim=1))

                sampled_probs = torch.mean(torch.stack(sampled_probs), dim=0)
                mc_metric.update(inputs.shape[0],
                                 torch.mean(torch.stack(sampled_losses)),
                                 sampled_probs, targets)

    if args.distributed:
        metric.sync()
        mc_metric.sync()
    print(f'Epoch {epoch} Test {metric}')
    wandb.log({
        'test/loss': metric.loss,
        'test/accuracy': metric.accuracy
    }, epoch)
    if args.opt == 'ivon':
        print(f'Epoch {epoch} MC Test {mc_metric}')
        wandb.log(
            {
                'test/mc_accuracy': mc_metric.accuracy,
                'test/mc_loss': mc_metric.loss
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
    dataset = create_dataset(args)

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
        if args.model_path is not None:
            model.load_state_dict(
                torch.load(args.model_path, map_location=args.device))
        model.to(args.device)
        if args.distributed:
            model = DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = create_model(args)
    for k, v in model.named_parameters():
        if 'bn' in k:
            setattr(v, 'disable_ivon', True)
            print(f'disable ivon for {k}')

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
                      rho=args.rho,
                      weight_decay=args.weight_decay)
            base_opt = opt.base_optimizer
    elif args.opt == 'ivon':
        opt = IVON(model.parameters(),
                   lr=args.lr,
                   data_size=len(dataset.train_dataset) * args.da_factor,
                   mc_samples=args.mc_samples,
                   momentum_grad=args.momentum_grad,
                   momentum_hess=args.momentum_hess,
                   prior_precision=args.prior_prec,
                   dampening=args.init_dp,
                   hess_init=args.hess_init,
                   world_size=args.world_size)
        base_opt = opt
    else:
        raise ValueError(f'Unknown optimizer {args.opt}')

    # ========== LEARNING RATE SCHEDULER ==========
    iters_per_epoch = len(dataset.train_loader)
    if args.lr_sche == 'cos':
        cos_epochs = args.epochs - args.cooldown_epochs
        pct_warm = args.warmup_steps / iters_per_epoch / cos_epochs
        lr_scheduler = OneCycleLR(base_opt,
                                  max_lr=args.lr,
                                  anneal_strategy=args.lr_sche,
                                  steps_per_epoch=len(dataset.train_loader),
                                  epochs=cos_epochs,
                                  pct_start=pct_warm,
                                  cycle_momentum=False)
        dp_scheduler = CosSche(args.dp, args.init_dp,
                               cos_epochs * iters_per_epoch, args.warmup_steps)
    elif args.lr_sche == 'step':
        milestones = [x * dataset.iters_per_epoch for x in args.lr_decay_epoch]
        lr_scheduler = MultiStepLR(opt, milestones, gamma=0.1)
        ratio = (args.dp / args.init_dp)**(1. / len(milestones))
        dp_scheduler = StepSche(args.init_dp, milestones, ratio=ratio)
        #dp_scheduler = CosSche(args.dp, args.init_dp,
        #                       args.epochs * iters_per_epoch, args.warmup_steps)
    else:
        raise ValueError(f'Unknown learning rate scheduler {args.lr_sche}')

    # ========== TRAINING ==========
    for e in range(args.epochs):
        train(e, dataset, model, opt, args)
        test(e, dataset, model, args)

    if args.rank == 0:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        #print(f'saving to {args.dir}/{timestamp}')
        #torch.save(model.state_dict(), f"{args.dir}/{timestamp}")
