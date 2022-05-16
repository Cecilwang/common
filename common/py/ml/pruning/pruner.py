from collections import OrderedDict
from itertools import islice
import math

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization
from torch.nn.utils.parametrize import remove_parametrizations

from common.py.ml.util.ifvp import IFVPs
from common.py.ml.util.util import to_vector, list_module


def percentile(data, percentage):
    k = int(1 + torch.round(percentage * (data.numel() - 1)).item())
    return data.view(-1).kthvalue(k).values.item()


class Mask(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.n = weight.numel()

    def forward(self, weight):
        return weight * self.mask

    def right_inverse(self, weight):
        return weight


class StaticMask(Mask):
    def __init__(self, weight):
        super().__init__(weight)
        self.mask = nn.Parameter(torch.ones_like(weight), requires_grad=False)


class ScoreMask(Mask):
    def __init__(self, weight, init_score):
        super().__init__(weight)
        self.scores = nn.Parameter(torch.empty_like(weight))
        if init_score == 'abs_magnitude':
            self.scores.data = torch.abs(weight)
        elif init_score == 'kaiming':
            if self.scores.ndim < 2:
                nn.init.uniform_(self.scores, a=-1.0, b=1.0)
            else:
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        else:
            raise NotImplementedError


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

    def __init__(self, weight, init_score, sparsity):
        super().__init__(weight, init_score)
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

    def __init__(self, weight, init_score, threshold):
        super().__init__(weight, init_score)
        self.threshold = threshold

    @property
    def mask(self):
        return self.Op.apply(self.scores, self.threshold)


class Prunner:
    def __init__(self, model, ignore, without_bias=False):
        self.model = model
        self.modules = list_module(
            model,
            condition=lambda x:
            (not isinstance(x, ignore)) and hasattr(x, 'weight'))
        self.without_bias = without_bias
        self.n = 0
        self._param = []
        self._mask = []
        print('Pruning Scope:')
        for k, x in self.modules.items():
            print(f'\t{k}.weight')
            self.device = x.weight.device
            self.n += x.weight.numel()
            register_parametrization(x, 'weight', StaticMask(x.weight))
            self._param.append(x.parametrizations.weight.original)
            self._mask.append(x.parametrizations.weight[0].mask)
            if not self.without_bias and x.bias is not None:
                print(f'\t{k}.bias')
                self.n += x.bias.numel()
                register_parametrization(x, 'bias', StaticMask(x.bias))
                self._param.append(x.parametrizations.bias.original)
                self._mask.append(x.parametrizations.bias[0].mask)

    def finalize(self):
        self._param = []
        self._mask = []
        for k, x in self.modules.items():
            remove_parametrizations(x, 'weight')
            if not self.without_bias and x.bias is not None:
                remove_parametrizations(x, 'bias')

    def dump(self):
        state = {}
        for k, v in self.model.state_dict().items():
            if 'parametrizations' in k:
                if 'original' in k:
                    term = k.split('.')
                    m = '.'.join(term[:-3])
                    p = term[-2]
                    state[f'{m}.{p}'] = getattr(self.modules[m], p).clone()
            else:
                state[k] = v.clone()
        return state

    @property
    def param(self):
        return to_vector(self._param)

    @param.setter
    def param(self, param):
        l = 0
        for x in self._param:
            r = l + x.numel()
            x.data = param[l:r].reshape(x.shape)
            l = r

    @property
    def mask(self):
        return to_vector(self._mask)

    @mask.setter
    def mask(self, mask):
        l = 0
        for x in self._mask:
            r = l + x.numel()
            x.data = mask[l:r].reshape(x.shape)
            l = r

    @property
    def grad(self):
        return to_vector([x.grad for x in self._param])

    @property
    def n_zero(self):
        return len((self.mask == 0.0).nonzero())

    @property
    def sparsity(self):
        return self.n_zero / self.n

    #def apply_mask(self):
    #    for x, m in zip(self._param, self._mask):
    #        x.data *= m.data

    def prune(self, sparsity):
        with torch.no_grad():
            n_pruned = int((sparsity - self.sparsity) * self.n)

            score = self.score()
            _, indices = torch.sort(score)
            threshold = score[indices[n_pruned]]
            print(f'sparsity {sparsity:.2f} threshold {threshold}')
            indices = indices[:n_pruned]
            indices, _ = torch.sort(indices)

            mask = self.mask
            mask[indices] = 0
            self.mask = mask
            return indices

    def draw(self):
        m = int(math.ceil(math.sqrt(self.n)))
        img = np.zeros(m * m)
        img[:self.n] = self.mask
        plt.imshow(img.reshape(m, m), cmap='hot')
        plt.colorbar()
        plt.show()
        img[:self.n] = self.param
        plt.imshow(img.reshape(m, m), cmap='hot')
        plt.colorbar()
        plt.show()


class Magnitude(Prunner):
    def __init__(self, model, ignore=tuple()):
        super().__init__(model, ignore)

    def score(self):
        return torch.abs(self.param).masked_fill(self.mask == 0.0,
                                                 float('inf'))


class OptimalBrainSurgeon(Prunner):
    def __init__(self, model, ignore=tuple(), block_size=-1, block_batch=-1):
        super().__init__(model, ignore)
        self.model = model
        if block_size == -1:
            self.block_size = self.n
        self.block_size = block_size
        if block_batch == -1:
            block_batch = math.ceil(self.n / self.block_size)
        self.block_batch = block_batch

    def calc_fisher(self, loader, n_batch, damping=1e-3):
        grads = []
        for inputs, targets in islice(loader, n_batch):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            nn.CrossEntropyLoss()(self.model(inputs), targets).backward()
            g = self.grad
            grads.append(g)
        grads = torch.vstack(grads)
        self.ifvp = IFVPs(grads, self.block_size, self.block_batch, damping)
        self.ifisher_diag = self.ifvp.diag()

    def score(self):
        score = self.param.pow(2) / self.ifisher_diag
        score = score.masked_fill(self.mask == 0.0, float('inf'))
        return score

    def pruning_direction(self, i):
        p = self.param
        tmp = torch.zeros_like(p)
        tmp[i] = -p[i] / self.ifisher_diag[i]
        d = self.ifvp(tmp)
        d *= self.mask
        d[i] = -p[i]
        return d

    def prune(self, sparsity):
        indices = super().prune(sparsity)
        self.param = self.param + self.pruning_direction(indices)


class Movement():
    def __init__(self,
                 model,
                 ignore=tuple(),
                 without_bias=False,
                 init_score='abs_magnitude'):
        self.modules = list_module(
            model,
            condition=lambda x:
            (not isinstance(x, ignore)) and hasattr(x, 'weight'))
        self.without_bias = without_bias
        self.n = 0
        self._sparsity = torch.tensor(0.0)
        self._threshold = torch.tensor(0.0)
        self._scores = []
        print('Pruning Scope:')
        for k, x in self.modules.items():
            print(f'\t{k}.weight')
            self.n += x.weight.numel()
            register_parametrization(
                x, 'weight',
                ThresholdMask(x.weight, init_score, self._threshold))
            self._scores.append(x.parametrizations.weight[0].scores)
            if not self.without_bias and x.bias is not None:
                print(f'\t{k}.bias')
                self.n += x.bias.numel()
                register_parametrization(
                    x, 'bias',
                    ThresholdMask(x.bias, init_score, self._threshold))
                self._scores.append(x.parametrizations.bias[0].scores)
        self.update_threshold()

    def finalize(self):
        self._scores = []
        for k, x in self.modules.items():
            remove_parametrizations(x, 'weight')
            if not self.without_bias and x.bias is not None:
                remove_parametrizations(x, 'bias')

    @property
    def scores(self):
        return to_vector(self._scores)

    def prune(self, sparsity):
        self._sparsity.data = torch.tensor(sparsity)
        self.update_threshold()

    def update_threshold(self):
        self._threshold.data = torch.tensor(
            percentile(self.scores, self._sparsity))
        print(f'sparsity {self._sparsity:.2f} threshold {self._threshold}')

    @property
    def n_zero(self):
        return self._sparsity * self.n

    @property
    def sparsity(self):
        return self._sparsity.item()
