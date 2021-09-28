import argparse
from datetime import datetime

import torch
import wandb
import asdfghjkl as asdl

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
    parser.add_argument("--wd", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--stat_intvl", type=int, default=10)
    parser.add_argument("--inv_intvl", type=int, default=100)

    return parser.parse_args()


def update_stat(model, inputs, targets):
    mgr = asdl.fisher_for_cross_entropy(model, [asdl.COV], [asdl.SHAPE_KRON],
                                        inputs=inputs,
                                        targets=targets)
    mgr.accumulate_matrices('kfac', smoothing_weight=0.05)


def calc_inverse_of_stat(A, B, args):
    with torch.no_grad():
        NA = A.shape[0]
        NB = B.shape[0]
        pi = torch.trace(A) / torch.trace(B) * NB / NA
        regA = torch.eye(NA).to(args.device) * torch.sqrt(args.damping * pi)
        regB = torch.eye(NB).to(args.device) * torch.sqrt(args.damping / pi)
        invB = (B + regB).inverse()
        invA = (A + regA).inverse()
    return invA, invB


def calc_inverse_of_model(model, args):
    for layer in model.children():
        if (hasattr(layer, 'cov')):
            invA, invB = calc_inverse_of_stat(layer.kfac_cov.kron.A,
                                              layer.kfac_cov.kron.B, args)
            setattr(layer, 'invA', invA)
            setattr(layer, 'invB', invB)


def precondition_layer(layer):
    g = layer.weight.grad
    invA = layer.invA
    invB = layer.invB

    g = g.view(invB.shape[1], -1)
    if layer.bias is not None:
        g = torch.cat([g, layer.bias.grad.data.view(-1, 1)], 1)

    g = invB @ g @ invA

    if layer.bias is not None:
        layer.weight.grad = g[:, :-1].view(layer.weight.grad.shape)
        layer.bias.grad = g[:, -1:].view(layer.bias.grad.shape)
    else:
        layer.weight.grad = g.view(layer.weight.grad.shape)


def precondition_model(model):
    for layer in model.children():
        if (hasattr(layer, 'cov')):
            precondition_layer(layer)


def train(model, loader, loss_fn, opt, metrics, device, epoch, args):
    model.train()
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if i % args.stat_intvl == 0:
            update_stat(model, inputs, targets)

        opt.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()

        if i % args.inv_intvl == 0:
            calc_inverse_of_model(model, args)

        precondition_model(model)
        opt.step()
        metrics += (output, targets)
        if i % args.log_intvl == 0 or i == len(loader) - 1:
            print("Epoch {} Train: {}".format(epoch, metrics))
    wandb.log({"train_loss": metrics[1](), "train_acc": metrics[2]()})


def test(model, loader, metrics, epoch, device):
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            metrics += (output, targets)
    wandb.log({"test_loss": metrics[1](), "test_acc": metrics[2]()})
    cprint("red")("Epoch {} Test: {}".format(epoch, metrics))


def main():
    args = parse_args()
    wandb.init(project="kfac")
    wandb.run.name = "asdl-kfac-{}-lr{}-wd{}-damping{}".format(
        args.model, args.lr, args.wd, args.damping)

    model, train_loader, test_loader, loss_fn = MNIST(**vars(args))
    model.to(args.device)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_M = Metrics([Progress(len(train_loader)), Loss(loss_fn), Accuracy()])
    test_M = Metrics([Progress(len(test_loader)), Loss(loss_fn), Accuracy()])
    for i in range(args.e):
        train(model, train_loader, loss_fn, opt, train_M, args.device, i, args)
        test(model, test_loader, test_M, i, args.device)


if __name__ == "__main__":
    main()
