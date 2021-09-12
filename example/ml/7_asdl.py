import argparse

import asdfghjkl as asdl
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="ASDL")
    return parser.parse_args()


def print_result(model, name='model'):
    print(model)
    for att in dir(model):
        if att.startswith('fisher_') or att.startswith('cov'):
            result = getattr(model, att)
            if result.data is not None:
                print('{}.data = \n\t{}'.format(att, result.data))
            if result.kron is not None:
                print('{}.kron.A = \n\t{}'.format(att, result.kron.A))
                print('{}.kron.B = \n\t{}'.format(att, result.kron.B))
            if result.unit is not None:
                print('{}.unit = \n\t{}'.format(att, result.unit))
            if result.diag is not None:
                print('{}.diag = \n\t{}'.format(att, result.diag.data))
    print('=' * 20)
    for c in model.children():
        print_result(c)


def main():
    args = parse_args()

    f1 = torch.nn.Linear(2, 2, bias=False)
    f2 = torch.nn.Linear(2, 2, bias=False)
    model = torch.nn.Sequential(f1, f2)
    inputs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    targets = torch.tensor([0, 1])
    asdl.fisher_for_cross_entropy(
        model,  #
        [
            asdl.FISHER_EXACT,
            asdl.FISHER_MC,
            asdl.COV,
        ],
        [
            asdl.SHAPE_FULL,
            asdl.SHAPE_BLOCK_DIAG,
            asdl.SHAPE_KRON,
            asdl.SHAPE_DIAG,
        ],
        inputs=inputs,
        targets=targets)
    print_result(model)


if __name__ == "__main__":
    main()
