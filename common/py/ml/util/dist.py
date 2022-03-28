import os

import torch
import wandb


def _setup_print_for_distributed(is_master):
    '''
    This function disables printing when not in master process
    '''
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _setup_wandb_for_distributed(is_master, args):
    if not is_master:

        def log(*args, **kwargs):
            pass

        wandb.log = log


def init_distributed_mode(args):
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        return

    args.distributed = True
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    print(f'distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.rank)
    _setup_print_for_distributed(args.rank == 0)
    _setup_wandb_for_distributed(args.rank == 0, args)
