import torch
import torch.nn as nn

from python import util


def _init(model):
    torch.manual_seed(19950214)
    for p in model.parameters():
        if len(p.data.shape) > 1:
            nn.init.xavier_uniform_(p.data)
        else:
            p.data = torch.zeros(p.data.shape)


def MLP(n_input, n_output):
    # Define
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_input, 128),
        nn.ReLU(),
        nn.Linear(128, n_output),
    )

    _init(model)
    return model


def CNN(n_input, n_output):
    # Define
    model = nn.Sequential(
        nn.Conv2d(n_input[0], 8, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(
            16 * (((n_input[1] - 2) // 2 - 2) // 2) *
            (((n_input[2] - 2) // 2 - 2) // 2), n_output),
    )

    _init(model)
    return model


def save(model, name):
    util.save({k: v.data for k, v in model.named_parameters()}, name)


def load(model, name):
    ps = util.load(name)
    nps = model.named_parameters()
    for k, v in nps:
        if k in ps:
            v.data = ps[k]
