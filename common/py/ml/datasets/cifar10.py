from torchvision import datasets
from torchvision import transforms

from common.py.ml.datasets.datasets import Dataset


class CIFAR10(Dataset):
    def __init__(self, args, **kwargs):
        self.num_classes = 10
        if args.estimate_mean_and_std:
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        self.img_size = 32
        self.train_dataset = datasets.CIFAR10(
            args.data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(self.img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]))
        self.val_dataset = datasets.CIFAR10(args.data_path,
                                            train=False,
                                            download=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    self.mean, self.std)
                                            ]))
        super().__init__(args, **kwargs)
