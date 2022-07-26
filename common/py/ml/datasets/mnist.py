from torchvision import datasets
from torchvision import transforms

from common.py.ml.datasets.datasets import Dataset


class MNIST(Dataset):
    def __init__(self, args, **kwargs):
        self.num_classes = 10
        if args.estimate_mean_and_std:
            self.mean = [0.1307]
            self.std = [0.3081]
        else:
            self.mean = [0.5]
            self.std = [0.5]
        self.img_size = 32
        transform = transforms.Compose([
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.train_dataset = datasets.MNIST(args.data_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
        self.val_dataset = datasets.MNIST(args.data_path,
                                          train=False,
                                          download=False,
                                          transform=transform)
        super().__init__(args, **kwargs)
