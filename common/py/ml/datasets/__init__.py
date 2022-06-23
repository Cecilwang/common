from common.py.ml.datasets.mnist import MNIST
from common.py.ml.datasets.imagenet import IMAGENET
from common.py.ml.datasets.cifar10 import CIFAR10


def define_dataset_arguments(parser):
    parser.add_argument('--dataset',
                        default='MNIST',
                        type=str,
                        choices=['IMAGENET', 'CIFAR10', 'MNIST'])
    parser.add_argument('--data-path', default='/tmp', type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--val-batch-size', default=2048, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--label-smoothing', default=0.1, type=float)


def create_dataset(args, **kwargs):
    if args.dataset == 'IMAGENET':
        dataset = IMAGENET(args, **kwargs)
    if args.dataset == 'CIFAR10':
        dataset = CIFAR10(args, **kwargs)
    elif args.dataset == 'MNIST':
        dataset = MNIST(args, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
    args.num_classes = dataset.num_classes
    return dataset
