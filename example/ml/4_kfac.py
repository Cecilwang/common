import argparse

import torch

from python.ml.kfac import classification_sampling
from python.ml.kfac import KFAC
from python.ml.kfac import EKFAC
from python.ml.util.metrics import Accuracy
from python.ml.util.metrics import Loss
from python.ml.util.metrics import Progress
from python.ml.util.metrics import Metrics
from python.util import cprint

from model import save
from problem import MNIST


def parse_args():
    parser = argparse.ArgumentParser(description="kfac")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--log_intvl", type=int, default=500)
    parser.add_argument("--cov_intvl", type=int, default=10)
    parser.add_argument("--inv_intvl", type=int, default=100)
    parser.add_argument("--data", type=str, default="./.data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opt",
                        type=str,
                        default="kfac",
                        choices=["kfac", "ekfac", "sgd"])
    parser.add_argument("--log", type=str, default="./.log")
    return parser.parse_args()


def train(model, loader, loss_fn, opt, metrics, device, epoch, args):
    model.train()
    for i, (input, target) in enumerate(loader):
        if isinstance(opt, (KFAC, EKFAC)) and opt.steps % opt.cov_intvl == 0:
            opt.hook_on = True
        input, target = input.to(device), target.to(device)
        opt.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        if isinstance(opt, (KFAC, EKFAC)) and opt.steps % opt.cov_intvl == 0:
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


def test(model, loader, metrics, epoch, device):
    model.eval()
    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            metrics += (output, target)
    cprint("red")("Epoch {} Test: {}".format(epoch, metrics))


def main():
    args = parse_args()

    model, train_loader, test_loader, loss = MNIST(**vars(args))
    model.to(args.device)

    if args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == "kfac":
        opt = KFAC(model.parameters(), args.lr, args.damping, args.cov_intvl,
                   args.inv_intvl)
        opt.register(model)
    elif args.opt == "ekfac":
        opt = EKFAC(model.parameters(), args.lr, args.damping, args.cov_intvl,
                    args.inv_intvl)
        opt.register(model)

    train_M = Metrics([Progress(len(train_loader)), Loss(loss), Accuracy()])
    test_M = Metrics([Progress(len(test_loader)), Loss(loss), Accuracy()])
    for i in range(args.e):
        save(model, args.log + "/{}-{}.pkl".format(args.opt, i))
        train(model, train_loader, loss, opt, train_M, args.device, i, args)
        test(model, test_loader, test_M, i, args.device)
    save(model, args.log + "/{}-{}.pkl".format(args.opt, args.e))


if __name__ == "__main__":
    main()
