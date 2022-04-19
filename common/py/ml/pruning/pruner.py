from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization


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


def percentile(data, percentage):
    k = int(1 + torch.round(percentage * (data.numel() - 1)).item())
    return data.view(-1).kthvalue(k).values.item()


class Mask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight):
        return weight * self.mask

    def right_inverse(self, weight):
        return weight


class ScoreMask(Mask):
    def __init__(self, scores):
        super().__init__()
        self.n = scores.numel()
        self.scores = scores.clone()
        #self.scores = nn.Parameter(torch.empty(weight.shape))
        #if self.scores.ndim < 2:
        #    nn.init.uniform_(self.scores, a=-1.0, b=1.0)
        #else:
        #    nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))


class TopKMask(ScoreMask):
    class Op(torch.autograd.Function):
        @staticmethod
        def forward(ctx, scores, sparsity):
            zeros = torch.zeros_like(scores).to(scores.device)
            ones = torch.ones_like(scores).to(scores.device)
            return torch.where(scores < percentile(scores, sparsity), zeros,
                               ones)

        @staticmethod
        def backward(ctx, g):
            return g, None

    def __init__(self, weight, sparsity):
        super().__init__(weight)
        self.sparsity = sparsity

    @property
    def mask(self):
        return self.Op.apply(self.scores, self.sparsity)


class ThresholdMask(ScoreMask):
    class Op(torch.autograd.Function):
        @staticmethod
        def forward(ctx, scores, threshold):
            zeros = torch.zeros_like(scores).to(scores.device)
            ones = torch.ones_like(scores).to(scores.device)
            return torch.where(scores < threshold, zeros, ones)

        @staticmethod
        def backward(ctx, g):
            return g, None

    def __init__(self, weight, threshold):
        super().__init__(weight)
        self.threshold = threshold

    @property
    def mask(self):
        return self.Op.apply(self.scores, self.threshold)


class Prunner:
    def __init__(self, model, ignore):
        self.modules = list_module(
            model,
            condition=lambda x:
            (not isinstance(x, ignore)) and hasattr(x, 'weight'))
        self.n = 0
        self._params = []
        self._masks = []
        print('Pruning Scope:')
        for k, x in self.modules.items():
            print(f'{k}')
            self.n += torch.numel(x.weight)
            x.register_buffer("weight_mask", torch.ones_like(x.weight))
            #x.weight.register_hook(lambda g: g * x.weight_mask)
            self._params.append(x.weight)
            self._masks.append(x.weight_mask)
            if x.bias is not None:
                self.n += torch.numel(x.bias)
                x.register_buffer("bias_mask", torch.ones_like(x.bias))
                #x.bias.register_hook(lambda g: g * x.bias_mask)
                self._params.append(x.bias)
                self._masks.append(x.bias_mask)

    @property
    def params(self):
        return to_vector(self._params)

    @property
    def masks(self):
        return to_vector(self._masks)

    @masks.setter
    def masks(self, masks):
        l = 0
        for x in self._masks:
            r = l + torch.numel(x)
            x.data = masks[l:r].reshape(x.shape)
            l = r

    @property
    def grad(self):
        return to_vector([x.grad for x in self._params])

    @property
    def n_zero(self):
        return len((self.masks == 0.0).nonzero())

    @property
    def sparsity(self):
        return self.n_zero / self.n

    def apply_mask(self):
        for x, m in zip(self._params, self._masks):
            x.data *= m.data

    def prune(self, sparsity):
        with torch.no_grad():
            n_pruned = int((sparsity - self.sparsity) * self.n)

            scores = self.scores()
            _, indices = torch.sort(scores)
            print(f'sparsity {sparsity} threshold {scores[indices[n_pruned]]}')
            indices = indices[:n_pruned]
            indices, _ = torch.sort(indices)

            masks = self.masks
            masks[indices] = 0
            self.masks = masks
            self.apply_mask()


class Magnitude(Prunner):
    def __init__(self, model, ignore=tuple()):
        super().__init__(model, ignore)

    def scores(self):
        return torch.abs(self.params).masked_fill(self.masks == 0.0,
                                                  float("inf"))


class Movement():
    def __init__(self, model, ignore=tuple(), without_bias=False):
        self.modules = list_module(
            model,
            condition=lambda x:
            (not isinstance(x, ignore)) and hasattr(x, 'weight'))
        self._sparsity = torch.tensor(0.0)
        self._threshold = torch.tensor(0.0)
        self._scores = []
        self.n = 0
        print('Pruning Scope:')
        for k, x in self.modules.items():
            print(f'{k}')
            self.n += x.weight.numel()
            register_parametrization(x, "weight",
                                     ThresholdMask(x.weight, self._threshold))
            self._scores.append(x.parametrizations.weight[0].scores)
            if not without_bias and x.bias is not None:
                self.n += x.bias.numel()
                register_parametrization(
                    x, "bias", ThresholdMask(x.bias, self._threshold))
                self._scores.append(x.parametrizations.bias[0].scores)
        self.update_threshold()

    @property
    def scores(self):
        return to_vector(self._scores)

    def prune(self, sparsity):
        self._sparsity.data = torch.tensor(sparsity)
        self.update_threshold()

    def update_threshold(self):
        self._threshold.data = torch.tensor(
            percentile(self.scores, self._sparsity))
        print(f'sparsity {self._sparsity} threshold {self._threshold}')

    @property
    def n_zero(self):
        return self._sparsity * self.n

    def apply_mask(self):
        pass

    @property
    def sparsity(self):
        return self._sparsity.item()
