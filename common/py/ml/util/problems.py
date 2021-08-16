import torch.nn as nn

from common.py.ml.util import models as M
from common.py.ml.util import datasets as D


def MNIST(**kargs):
    model = {
        "cnn": M.CNN((1, 28, 28), 10),
        "mlp": M.MLP(784, 10)
    }[kargs["model"]]
    train_loader, test_loader = D.MNIST(**kargs)
    return model, train_loader, test_loader, nn.CrossEntropyLoss()
