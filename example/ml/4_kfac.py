import argparse

import torch

from common.py.ml.kfac import classification_sampling
from common.py.ml.kfac import KFAC
from common.py.ml.datasets import MNIST
from common.py.ml.models import MNISTToy, ViT
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric
from common.py.util import cprint


def parse_args():
    parser = argparse.ArgumentParser(description="kfac")
    parser.add_argument('--data-path', default='/tmp', type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument('--model',
                        default='MNISTToy',
                        type=str,
                        choices=['MNISTToy', 'ViT'])

    subparsers = parser.add_subparsers()

    parser_sgd = subparsers.add_parser("sgd", help="SGD")
    parser_sgd.set_defaults(opt="sgd")
    parser_sgd.add_argument("--lr", type=float, default=1e-3)
    parser_sgd.add_argument("--wd", type=float, default=0.01)

    parser_kfac = subparsers.add_parser("kfac", help="K-FAC")
    parser_kfac.set_defaults(opt="kfac")
    parser_kfac.add_argument("--lr", type=float, default=1e-2)
    parser_kfac.add_argument("--damping", type=float, default=1.0)
    parser_kfac.add_argument("--cov_intvl", type=int, default=10)
    parser_kfac.add_argument("--inv_intvl", type=int, default=100)

    return parser.parse_args()


def train(epoch, dataset, model, criterion, opt, args):
    model.train()
    dataset.train()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    metric = Metric(args.device)

    for i, (inputs, targets) in enumerate(dataset.loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        if isinstance(opt, (KFAC, )) and opt.steps % opt.cov_intvl == 0:
            opt.hook_on = True
        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if isinstance(opt, (KFAC, )) and opt.steps % opt.cov_intvl == 0:
            with torch.no_grad():
                sampled_y = classification_sampling(outputs)
            criterion(outputs, sampled_y).backward(retain_graph=True)
            opt.zero_grad()
            opt.hook_on = False
        loss.backward()
        opt.step()
        metric.update(inputs.shape[0], loss, outputs, targets)
    print("Epoch {} Train: {}".format(epoch, metric))


def test(epoch, dataset, model, criterion, args):
    model.eval()
    dataset.eval()
    metric = Metric(args.device)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataset.loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metric.update(inputs.shape[0], loss, outputs, targets)
    cprint("red")("Epoch {} Test: {}".format(epoch, metric))


def main():
    args = parse_args()
    init_distributed_mode(args)
    dataset = MNIST(args)
    criterion = torch.nn.CrossEntropyLoss()
    if args.model == "MNISTToy":
        model = MNISTToy()
    elif args.model == "ViT":
        model = ViT(1,
                    32,
                    32,
                    256,
                    3,
                    10,
                    patch_size=16,
                    attention_dropout=0.9,
                    n_heads=8,
                    expansion=4,
                    forward_dropout=0.9,
                    dropout=0.9)
    model.to(args.device)

    if args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.wd)
    elif args.opt == "kfac":
        opt = KFAC(model.parameters(), args.lr, args.damping, args.cov_intvl,
                   args.inv_intvl)
        opt.register(model)

    for e in range(args.epochs):
        train(e, dataset, model, criterion, opt, args)
        test(e, dataset, model, criterion, args)


if __name__ == "__main__":
    main()
