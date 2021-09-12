import argparse
from datetime import datetime
import math
import operator

import einops
import matplotlib.pyplot as plt
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import wandb

from common.py.ml.kfac import classification_sampling
from common.py.ml.kfac import KFAC
from common.py.ml.util.metrics import Accuracy
from common.py.ml.util.metrics import Loss
from common.py.ml.util.metrics import Progress
from common.py.ml.util.metrics import Metrics
from common.py.ml.util.models import save
from common.py.ml.util.datasets import CIFAR10
from common.py.util import cprint


def calc_mean_and_stddev(loader):

    class Stats(PIL.ImageStat.Stat):

        def __add__(self, other):
            return Stats(list(map(operator.add, self.h, other.h)))

    stats = None
    for x, _ in loader:
        for img in x:
            if stats is None:
                stats = Stats(transforms.ToPILImage()(img))
            else:
                stats += Stats(transforms.ToPILImage()(img))
    print("mean:{}, std:{}".format(stats.mean, stats.stddev))
    return stats


def show_patches(args, img):
    tpl = "c (h ph) (w pw) -> h w c ph pw"
    patches = einops.rearrange(img, tpl, ph=args.patch_size, pw=args.patch_size)
    nh, nw = patches.shape[0], patches.shape[1]

    ax = plt.subplot2grid((nh, nw * 2), (0, 0), rowspan=nh, colspan=nw)
    ax.imshow(transforms.ToPILImage()(img))
    for h in range(nh):
        for w in range(nw):
            ax = plt.subplot2grid((nw, nw * 2), (h, nw + w))
            ax.imshow(transforms.ToPILImage()(patches[h][w]))
    plt.show()


class PatchEmbedding(nn.Module):

    def __init__(self, n_channels, h, w, n_embedding, patch_size=16, **_):
        super().__init__()
        self.conv2d = nn.Conv2d(n_channels,
                                n_embedding,
                                kernel_size=patch_size,
                                stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, n_embedding))
        self.positions = nn.Parameter(
            torch.randn((h // patch_size) * (w // patch_size) + 1, n_embedding))

    def forward(self, x):
        x = self.conv2d(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        cls_tokens = einops.repeat(self.cls_token, "n e -> b n e", b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class Attention(nn.Module):

    def __init__(self, n_embedding, n_heads=8, attention_dropout=0.0, **_):
        super().__init__()
        self.n_embedding = n_embedding
        self.n_heads = n_heads
        self.keys = nn.Linear(self.n_embedding, self.n_embedding)
        self.queries = nn.Linear(self.n_embedding, self.n_embedding)
        self.values = nn.Linear(self.n_embedding, self.n_embedding)
        self.dropout = nn.Dropout(attention_dropout)
        self.linear = nn.Linear(self.n_embedding, self.n_embedding)

    def forward(self, x, mask=None):
        tpl = "b n (h d) -> b h n d"
        queries = einops.rearrange(self.queries(x), tpl, h=self.n_heads)
        keys = einops.rearrange(self.keys(x), tpl, h=self.n_heads)
        values = einops.rearrange(self.values(x), tpl, h=self.n_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            energy.mask_fill(~mask, torch.finfo(torch.float32).min)

        attention = F.softmax(energy, dim=-1) / math.sqrt(self.n_embedding)
        attention = self.dropout(attention)
        x = torch.einsum("bhqk, bhkd -> bhqd", attention, values)
        x = einops.rearrange(x, "b h n d -> b n (h d)")
        x = self.linear(x)
        return x


class ResidualAdd(nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x, **kwargs):
        return x + self.f(x, **kwargs)


class FeedForward(nn.Sequential):

    def __init__(self, n_embedding, expansion=4, forward_dropout=0.0, **_):
        super().__init__(
            nn.Linear(n_embedding, expansion * n_embedding),
            nn.GELU(),
            nn.Dropout(forward_dropout),
            nn.Linear(expansion * n_embedding, n_embedding),
        )


class TransformerEncoderBlock(nn.Sequential):

    def __init__(self, n_embedding, dropout=0.0, **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(n_embedding),
                              Attention(n_embedding, **kwargs),
                              nn.Dropout(dropout))),
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(n_embedding),
                              FeedForward(n_embedding, **kwargs),
                              nn.Dropout(dropout))))


class TransformerEncoder(nn.Sequential):

    def __init__(self, depth, **kwargs):
        super().__init__(
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):

    def __init__(self, n_embedding, n_classes):
        super().__init__(nn.LayerNorm(n_embedding),
                         nn.Linear(n_embedding, n_classes))

    def forward(self, x):
        return super().forward(x.mean(1))


class ViT(nn.Sequential):

    def __init__(self, n_channels, h, w, n_embedding, depth, n_classes,
                 **kwargs):
        super().__init__(
            PatchEmbedding(n_channels, h, w, n_embedding, **kwargs),
            TransformerEncoder(depth, n_embedding=n_embedding, **kwargs),
            ClassificationHead(n_embedding, n_classes))


def parse_args():
    parser = argparse.ArgumentParser(description="ViT")

    parser.add_argument("--data", type=str, default="./.data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)

    parser.add_argument("--n_embedding", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)

    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--log_intvl", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log", type=str, default="./.log")

    subparsers = parser.add_subparsers()

    parser_sgd = subparsers.add_parser("sgd", help="SGD")
    parser_sgd.set_defaults(opt="sgd")
    parser_sgd.add_argument("--lr", type=float, default=1e-3)
    parser_sgd.add_argument("--damping", type=float, default=0.01)

    parser_kfac = subparsers.add_parser("kfac", help="K-FAC")
    parser_kfac.set_defaults(opt="kfac")
    parser_kfac.add_argument("--lr", type=float, default=1e-2)
    parser_kfac.add_argument("--damping", type=float, default=1.0)
    parser_kfac.add_argument("--cov_intvl", type=int, default=10)
    parser_kfac.add_argument("--inv_intvl", type=int, default=100)

    return parser.parse_args()


def train(model, loader, loss_fn, opt, metrics, device, epoch, args):
    model.train()
    for i, (input, target) in enumerate(loader):
        if isinstance(opt, (KFAC,)) and opt.steps % opt.cov_intvl == 0:
            opt.hook_on = True
        input, target = input.to(device), target.to(device)
        opt.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        if isinstance(opt, (KFAC,)) and opt.steps % opt.cov_intvl == 0:
            with torch.no_grad():
                sampled_y = classification_sampling(output)
            loss_fn(output, sampled_y).backward(retain_graph=True)
            opt.zero_grad()
            opt.hook_on = False
        loss.backward()
        opt.step()
        metrics += (output, target)
        if i % args.log_intvl == 0 or i == len(loader) - 1:
            print("Epoch {} Train: {}".format(epoch, metrics))
    wandb.log({"train_loss": metrics[1](), "train_acc": metrics[2]()})


def test(model, loader, metrics, epoch, device):
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            metrics += (output, target)
    wandb.log({"test_loss": metrics[1](), "test_acc": metrics[2]()})
    cprint("red")("Epoch {} Test: {}".format(epoch, metrics))


def main():
    args = parse_args()

    wandb.init(project="ViT")
    wandb.run.name = "{}-{}-{}".format(args.opt, args.lr, datetime.now())

    train_loader, test_loader, classes = CIFAR10(**vars(args))

    #calc_mean_and_stddev(train_loader)
    #show_patches(args, train_loader.__iter__().next()[0][0])

    model = ViT(3,
                32,
                32,
                args.n_embedding,
                args.depth,
                10,
                patch_size=args.patch_size,
                attention_dropout=0.9,
                n_heads=8,
                expansion=4,
                forward_dropout=0.9,
                dropout=0.9)
    model.to(args.device)

    loss_fn = nn.CrossEntropyLoss()

    if args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.damping)
    elif args.opt == "kfac":
        opt = KFAC(model.parameters(), args.lr, args.damping, args.cov_intvl,
                   args.inv_intvl)
        opt.register(model)

    train_M = Metrics([Progress(len(train_loader)), Loss(loss_fn), Accuracy()])
    test_M = Metrics([Progress(len(test_loader)), Loss(loss_fn), Accuracy()])
    for i in range(args.e):
        save(model, args.log + "/{}-{}.pkl".format(args.opt, i))
        train(model, train_loader, loss_fn, opt, train_M, args.device, i, args)
        test(model, test_loader, test_M, i, args.device)
    save(model, args.log + "/{}-{}.pkl".format(args.opt, args.e))


if __name__ == "__main__":
    main()
