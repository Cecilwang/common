from torchvision import datasets
from torchvision import transforms

from common.py.ml.datasets.datasets import Dataset


class CIFAR100(Dataset):
    def __init__(self, args, **kwargs):
        self.num_classes = 100
        if args.estimate_mean_and_std:
            self.mean = [
                0.5070588235294118, 0.48666666666666664, 0.4407843137254902
            ]
            self.std = [
                0.26745098039215687, 0.2564705882352941, 0.27607843137254906
            ]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        self.img_size = 32
        self.train_dataset = datasets.CIFAR100(
            args.data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(self.img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]))
        self.val_dataset = datasets.CIFAR100(args.data_path,
                                             train=False,
                                             download=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     self.mean, self.std)
                                             ]))
        super().__init__(args, **kwargs)
