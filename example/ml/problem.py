import torch.nn as nn

import model
import dataset


def MNIST(**kargs):
    mlp = model.MLP(784, 10)
    train_loader, test_loader = dataset.MNIST(**kargs)
    return mlp, train_loader, test_loader, nn.CrossEntropyLoss()
