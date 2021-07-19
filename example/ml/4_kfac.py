import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from python.ml.kfac import KFAC
from python.ml.util.metrics import Accuracy
from python.ml.util.metrics import Loss
from python.ml.util.metrics import Progress
from python.ml.util.metrics import Metrics
from python.util import cprint


def parse_args():
    parser = argparse.ArgumentParser(description="kfac")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--intvl", type=int, default=500)
    parser.add_argument("--data_dir", type=str, default="./.data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opt",
                        type=str,
                        default="kfac",
                        choices=["kfac", "sgd"])
    return parser.parse_args()


def get_data(args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    train_datasets = datasets.MNIST(args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform)
    train_loader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=args.batch)
    test_datasets = datasets.MNIST(args.data_dir,
                                   train=False,
                                   transform=transform)
    test_loader = torch.utils.data.DataLoader(test_datasets,
                                              batch_size=args.batch)
    return train_loader, test_loader


def train(model, loader, loss, opt, metrics, device, epoch, args):
    model.train()
    for i, (input, target) in enumerate(loader):
        input, target = input.to(device), target.to(device)
        input = torch.flatten(input, start_dim=1)
        opt.zero_grad()
        output = model(input)
        loss(output, target).backward()
        opt.step()
        metrics += (output, target)
        if i % args.intvl == 0 or i == len(loader) - 1:
            print("Epoch {} Train: {}".format(epoch, metrics))


def test(model, loader, metrics, epoch, device):
    model.eval()
    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)
            input = torch.flatten(input, start_dim=1)
            output = model(input)
            metrics += (output, target)
    cprint("red")("Epoch {} Test: {}".format(epoch, metrics))


def main():
    args = parse_args()
    train_loader, test_loader = get_data(args)
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                          nn.Linear(128, 10)).to(args.device)
    loss = nn.CrossEntropyLoss()
    if args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == "kfac":
        opt = KFAC(model.parameters(), args.lr, args.damping)
        opt.register((model, loss))
    train_M = Metrics([Progress(len(train_loader)), Loss(loss), Accuracy()])
    test_M = Metrics([Progress(len(test_loader)), Loss(loss), Accuracy()])
    for i in range(args.e):
        train(model, train_loader, loss, opt, train_M, args.device, i, args)
        test(model, test_loader, test_M, i, args.device)


if __name__ == "__main__":
    main()
