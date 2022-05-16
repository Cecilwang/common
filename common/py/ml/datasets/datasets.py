import PIL

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


def calc_mean_and_stddev(loader):
    class Stats(PIL.ImageStat.Stat):
        def __add__(self, other):
            return Stats(list(map(operator.add, self.h, other.h)))

    stats = None
    for x, _ in loader:
        for img in x:
            if stats is None:
                stats = Stats(transforms.ToPILImage()(img))
            else:
                stats += Stats(transforms.ToPILImage()(img))
    print("mean:{}, std:{}".format(stats.mean, stats.stddev))
    return stats


class Dataset(object):
    def __init__(self, args, batch_size=None):
        self.batch_size = args.batch_size if batch_size is None else batch_size
        self.val_batch_size = args.val_batch_size
        self.num_workers = 4
        self.pin_memory = True
        if args.distributed:
            self.train_sampler = DistributedSampler(self.train_dataset)
        else:
            if args.shuffle:
                self.train_sampler = RandomSampler(self.train_dataset)
            else:
                self.train_sampler = SequentialSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=4,
                                       pin_memory=True)
        if args.distributed:
            self.val_sampler = DistributedSampler(self.val_dataset)
        else:
            self.val_sampler = RandomSampler(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.val_batch_size,
                                     sampler=self.val_sampler,
                                     num_workers=4,
                                     pin_memory=True)
        self.sampler = None
        self.loader = None
        self._criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing)

    def train(self):
        self.sampler = self.train_sampler
        self.loader = self.train_loader

    def eval(self):
        self.sampler = self.val_sampler
        self.loader = self.val_loader

    def criterion(self, *args, **kwargs):
        return self._criterion(*args, **kwargs)
