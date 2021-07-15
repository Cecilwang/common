import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms


class cprint:

    def __init__(self, color=None):
        self.color = {"red": "\033[91m"}.get(color, None)
        self.end = "\033[0m"

    def __call__(self, str):
        if self.color:
            print(self.color + str + self.end)
        else:
            print(str)


class Metric(object):

    def __init__(self):
        self.name = "none"
        self.n = 0
        self.val = 0

    def calc(self, output, target):
        raise NotImplementedError

    def __iadd__(self, update):
        n, val = self.calc(update[0], update[1])
        self.n += n
        self.val += val
        return self

    def __str__(self):
        return "{}: {}".format(self.name, self.val / self.n)


class Progress(Metric):

    def __init__(self, n):
        Metric.__init__(self)
        self.name = "Progress"
        self.n = n

    def calc(self, output, target):
        return 0, 1

    def __str__(self):
        width = len(str(self.n))
        return "{{}}:{{:>{0}}}/{{}}({{:3.0f}}%)".format(width).format(
            self.name, self.val, self.n, 100. * self.val / self.n)


class Loss(Metric):

    def __init__(self, loss):
        Metric.__init__(self)
        self.name = "Loss"
        self.loss = loss

    def calc(self, output, target):
        return len(output), self.loss(output, target) * len(output)

    def __str__(self):
        return "{}: {:.5f}".format(self.name, self.val / self.n)


class Accuracy(Metric):

    def __init__(self):
        Metric.__init__(self)
        self.name = "Accuracy"

    def calc(self, output, target):
        return len(output), output.argmax(dim=1).eq(target).sum().item()

    def __str__(self):
        return "{}: {:.0f}%".format(self.name, 100. * self.val / self.n)


class Metrics(object):

    def __init__(self, metrics):
        self.metrics = metrics

    def __iadd__(self, update):
        for x in self.metrics:
            x += update
        return self

    def __str__(self):
        return ", ".join([str(x) for x in self.metrics])


def parse_args():
    parser = argparse.ArgumentParser(description="kfac")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--e", type=int, default=20)
    parser.add_argument("--intvl", type=int, default=500)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cpu")
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
    model = nn.Sequential(nn.Linear(784, 10)).to(args.device)
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    train_M = Metrics([Progress(len(train_loader)), Loss(loss), Accuracy()])
    test_M = Metrics([Progress(len(test_loader)), Loss(loss), Accuracy()])
    for i in range(args.e):
        train(model, train_loader, loss, opt, train_M, args.device, i, args)
        test(model, test_loader, test_M, i, args.device)


if __name__ == "__main__":
    main()
