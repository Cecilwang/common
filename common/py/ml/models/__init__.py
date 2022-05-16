import torch

from common.py.ml.models.mnist_toy import MNISTToy
from common.py.ml.models.vit import ViT


def define_model_arguments(parser):
    parser.add_argument('--model',
                        default='MNISTToy',
                        type=str,
                        choices=['resnet50', 'MNISTToy'])
    parser.add_argument('--model-path',
                        default='example/ml/MNISTToy',
                        type=str)


def create_model(args):
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=dataset.num_classes)
    elif args.model == 'MNISTToy':
        model = MNISTToy()
    else:
        raise ValueError(f'Unknown model {args.model}')
    if hasattr(args, 'model_path') and args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    return model
