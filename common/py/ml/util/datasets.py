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


def CIFAR10(**kargs):
    data = kargs.get("data", "./.data")
    batch_size = kargs.get("batch_size", 32)
    shuffle = kargs.get("shuffle", True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(
        #    (125.3069180468 / 255, 122.950394140 / 255, 113.8653831835 / 255),
        #    (62.99321927813 / 255, 62.0887076400 / 255, 66.70489964063 / 255))
    ])

    train_datasets = datasets.CIFAR10(data,
                                      train=True,
                                      download=True,
                                      transform=transform)
    train_loader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    test_datasets = datasets.CIFAR10(data, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_datasets,
                                              batch_size=batch_size)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
               "ship", "truck")
    return train_loader, test_loader, classes
