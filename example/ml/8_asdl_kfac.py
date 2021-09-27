import argparse
from datetime import datetime

import torch
import wandb

from common.py.ml.util.metrics import Accuracy
from common.py.ml.util.metrics import Loss
from common.py.ml.util.metrics import Progress
from common.py.ml.util.metrics import Metrics
from common.py.ml.util.models import save
from common.py.ml.util.problems import MNIST
from common.py.util import cprint


def parse_args():
    parser = argparse.ArgumentParser(description="kfac")
    parser.add_argument("--model",
                        type=str,
                        default="cnn",
                        choices=["mlp", "cnn"])
    parser.add_argument("--data", type=str, default="./.data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--log_intvl", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log", type=str, default="./.log")

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--cov_intvl", type=int, default=10)
    parser.add_argument("--inv_intvl", type=int, default=100)

    return parser.parse_args()


def train(model, loader, loss_fn, opt, metrics, device, epoch, args):
    model.train()
    for i, (input, target) in enumerate(loader):
        input, target = input.to(device), target.to(device)
        opt.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
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
    wandb.init(project="kfac")
    wandb.run.name = "asdl-{}-{}".format(args.model, args.lr)

    model, train_loader, test_loader, loss_fn = MNIST(**vars(args))
    model.to(args.device)

    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr,
                          weight_decay=args.damping)

    train_M = Metrics([Progress(len(train_loader)), Loss(loss_fn), Accuracy()])
    test_M = Metrics([Progress(len(test_loader)), Loss(loss_fn), Accuracy()])
    for i in range(args.e):
        train(model, train_loader, loss_fn, opt, train_M, args.device, i, args)
        test(model, test_loader, test_M, i, args.device)


if __name__ == "__main__":
    main()
