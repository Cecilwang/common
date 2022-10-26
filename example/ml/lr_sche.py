import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from common.py.ml.models import define_model_arguments, create_model
from common.py.ml.util.dist import init_distributed_mode

import argparse

parser = argparse.ArgumentParser(description='lr-sche')
define_model_arguments(parser)
parser.add_argument('--device', default='cpu', type=str)
args = parser.parse_args()
init_distributed_mode(args)

model = create_model(args)
opt = torch.optim.SGD(model.parameters(), lr=1)

lr_warmup_sche = LinearLR(opt, 0.0001, total_iters=10)
lr_step_scheduler = MultiStepLR(opt, [1, 5], gamma=0.1)
lr_scheduler = SequentialLR(opt, [lr_warmup_sche, lr_step_scheduler], [10])
for epoch in range(20):
    lr_scheduler.step()
    lr = opt.param_groups[0]['lr']
    print(epoch, lr)
