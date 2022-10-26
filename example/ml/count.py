import torch

from common.py.ml.models import define_model_arguments, create_model
from common.py.ml.util.dist import init_distributed_mode


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero_parameters(model):
    return sum(
        torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)


import argparse

parser = argparse.ArgumentParser(description='count')
define_model_arguments(parser)
parser.add_argument('--device', default='cpu', type=str)
args = parser.parse_args()
init_distributed_mode(args)

model = create_model(args)
for p in model.parameters():
    if p.requires_grad:
        print(p.name)
        print(p.numel())
print(count_parameters(model))
print(count_nonzero_parameters(model))
