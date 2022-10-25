from common.py.ml.pruning.pruner import Magnitude
from common.py.ml.pruning.pruner import OptimalBrainSurgeon
from common.py.ml.pruning.pruner import Movement


def define_pruning_arguments(parser):
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--pruning_epochs', nargs='+', type=int, default=[2, 5])

    subparsers = parser.add_subparsers()
    parser_magnitude = subparsers.add_parser('magnitude', help='magnitude')
    parser_magnitude.set_defaults(pruner='magnitude')

    parser_movement = subparsers.add_parser('movement', help='movement')
    parser_movement.set_defaults(pruner='movement')
    parser_movement.add_argument('--init_score',
                                 type=str,
                                 default='abs_magnitude',
                                 choices=['abs_magnitude', 'kaiming'])

    parser_obs = subparsers.add_parser('obs', help='optimal brain surgeon')
    parser_obs.set_defaults(pruner='obs')
    parser_obs.add_argument('--fisher_batch_size',
                            type=int,
                            default=32,
                            help='batch size of fisher dataset')
    parser_obs.add_argument('--n_batch',
                            type=int,
                            default=64,
                            help='how many batches are used to compute fisher')
    parser_obs.add_argument('--n_recompute',
                            type=int,
                            default=16,
                            help='how many times to recompute fisher')
    parser_obs.add_argument('--damping', type=float, default=1e-4)
    parser_obs.add_argument('--block_size', type=int, default=128)
    parser_obs.add_argument('--block_batch', type=int, default=10000)
    parser_obs.add_argument(
        '--kfac_fast_inv',
        dest='kfac_fast_inv',
        action='store_true',
    )
    parser_obs.add_argument(
        '--layer_normalize',
        dest='layer_normalize',
        action='store_true',
    )


def create_pruner(args, model, ignore=list()):
    if args.pruner == 'magnitude':
        pruner = Magnitude(model, ignore)
    elif args.pruner == 'movement':
        pruner = Movement(model, ignore, init_score=args.init_score)
    elif args.pruner == 'obs':
        pruner = OptimalBrainSurgeon(model,
                                     ignore,
                                     block_size=args.block_size,
                                     block_batch=args.block_batch)
    return pruner
