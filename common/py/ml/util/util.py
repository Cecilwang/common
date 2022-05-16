from collections import OrderedDict

from torch import nn


def to_vector(parameters):
    return nn.utils.parameters_to_vector(parameters)


def list_module(module, prefix='', condition=lambda _: True):
    modules = OrderedDict()
    has_children = False
    for name, x in module.named_children():
        has_children = True
        new_prefix = prefix + ('' if prefix == '' else '.') + name
        modules.update(list_module(x, new_prefix, condition))
    if not has_children and condition(module):
        modules[prefix] = module
    return modules
