import torch

from common.py.ml.models.mnist_toy import MNISTToy
from common.py.ml.models.vit import ViT


def define_model_arguments(parser):
    parser.add_argument('--model',
                        default='MNISTToy',
                        type=str,
                        choices=['resnet50', 'resnet18', 'MNISTToy'])
    parser.add_argument('--model-path', default=None, type=str)


def create_model(args, model_path=None):
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=args.num_classes)
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(num_classes=args.num_classes)
    elif args.model == 'MNISTToy':
        model = MNISTToy()
    else:
        raise ValueError(f'Unknown model {args.model}')
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    elif hasattr(args, 'model_path') and args.model_path is not None:
        model.load_state_dict(
            torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    return model
