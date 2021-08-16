import torch.nn as nn

import model as M
import dataset


def MNIST(**kargs):
    model = {
        "cnn": M.CNN((1, 28, 28), 10),
        "mlp": M.MLP(784, 10)
    }[kargs["model"]]
    train_loader, test_loader = dataset.MNIST(**kargs)
    return model, train_loader, test_loader, nn.CrossEntropyLoss()
