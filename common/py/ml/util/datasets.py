import torch
from torchvision import datasets
from torchvision import transforms


def MNIST(**kargs):
    data = kargs.get("data", "./.data")
    batch_size = kargs.get("batch_size", 32)
    shuffle = kargs.get("shuffle", True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_datasets = datasets.MNIST(data,
                                    train=True,
                                    download=True,
                                    transform=transform)
    train_loader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    test_datasets = datasets.MNIST(data, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_datasets,
                                              batch_size=batch_size)
    return train_loader, test_loader
