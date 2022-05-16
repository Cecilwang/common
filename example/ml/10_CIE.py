from pathlib import Path
import os

import torch
from torch import nn

from common.py.ml.datasets import define_dataset_arguments, create_dataset
from common.py.ml.models import define_model_arguments, create_model
from common.py.ml.util.dist import init_distributed_mode
from common.py.ml.util.metrics import Metric


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='CIE')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    define_dataset_arguments(parser)
    define_model_arguments(parser)
    parser.add_argument('--compressed-model-path',
                        nargs='+',
                        type=str,
                        default=[
                            'example/ml/data/CompressedMNISTToy.map.92.1',
                            'example/ml/data/CompressedMNISTToy.map.92.2',
                            'example/ml/data/CompressedMNISTToy.map.92.3',
                            'example/ml/data/CompressedMNISTToy.map.92.4',
                            'example/ml/data/CompressedMNISTToy.map.92.5',
                        ])

    return parser.parse_args()


def test(epoch, dataset, model, args, prefix=''):
    dataset.eval()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.eval()

    metric = Metric(args.device)

    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(dataset.loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = dataset.criterion(outputs, targets)

            metric.update(inputs.shape[0], loss, outputs, targets)

    if args.distributed:
        metric.sync()
    print(f'Epoch {epoch} {prefix} Test {metric}')
    return metric.accuracy


if __name__ == '__main__':
    args = parse_args()
    args.name = f'{args.dataset}/{args.model}/CIE/{args.name}'
    args.dir = f'{args.dir}/{args.name}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    print(args)

    dataset = create_dataset(args)
    model = create_model(args)
    test(0, dataset, model, args, 'non-compressed')
    for i, x in enumerate(args.compressed_model_path):
        model = create_model(args, model_path=x)
        test(0, dataset, model, args, f'compressed {i}')
