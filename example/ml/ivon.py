from typing import Callable, Tuple, Optional, Sequence
from contextlib import contextmanager
import torch
from torch import Tensor
from torch.optim import Optimizer


ClosureType = Callable[[], Tuple[Tensor, Tensor]]


class VON(Optimizer):
    def __init__(
            self, params, lr, data_size: int, mc_samples: int = 1,
            momentum_grad: float = 0.9, momentum_hess: Optional[float] = None,
            prior_precision: float = 1.0, dampening: float = 0.0,
            hess_init: Optional[float] = None):
        assert lr > 0.0
        assert data_size >= 1
        assert mc_samples >= 1
        assert prior_precision > 0.0
        assert dampening >= 0.0
        if momentum_hess is None:
            momentum_hess = 1.0 - lr  # default follows theoretical derivation
        self.mc_samples = mc_samples
        defaults = dict(
            lr=lr, data_size=data_size, momentum_grad=momentum_grad,
            momentum_hess=momentum_hess, prior_precision=prior_precision,
            dampening=dampening)
        super().__init__(params, defaults)
        self._init_momentum_buffers(hess_init)
        self._reset_param_and_grad_samples()

    def _init_momentum_buffers(self, hess_init):
        for group in self.param_groups:
            data_size = group['data_size']
            prior_precision = group['prior_precision']
            dampening = group['dampening']
            if hess_init is None:
                hess_init_val = dampening + prior_precision / float(data_size)
            else:
                hess_init_val = hess_init
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['momentum_grad_buffer'] = torch.zeros_like(p)
                    self.state[p]['momentum_hess_buffer'] = torch.full_like(
                        p, hess_init_val)

    def _reset_param_and_grad_samples(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_samples'] = []
                    self.state[p]['grad_samples'] = []

    @torch.no_grad()
    def step(self, closure: ClosureType = None):
        if closure is None:
            raise ValueError('VON optimizer requires closure function.')

        self._stash_param_averages()

        losses = []
        outputs = []

        for _ in range(self.mc_samples):
            self._sample_weight_and_collect()
            with torch.enable_grad():
                loss, output = closure()
            losses.append(loss.detach())
            outputs.append(output.detach())
            self._collect_grad_samples()

        self._update()

        self._restore_param_averages()
        self._reset_param_and_grad_samples()
        avg_loss = torch.mean(torch.stack(losses, dim=0), dim=0)
        avg_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return avg_loss, avg_output

    def _stash_param_averages(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_average'] = p.data

    def _sample_weight_and_collect(self):
        for group in self.param_groups:
            n = group['data_size']
            for p in group['params']:
                if p.requires_grad:
                    m_hess = self.state[p]['momentum_hess_buffer']
                    p_avg = self.state[p]['param_average']
                    normal_sample = torch.randn_like(p)
                    p_sample = normal_sample * torch.rsqrt(n * m_hess) + p_avg
                    p.data = p_sample
                    self.state[p]['param_samples'].append(p_sample)

    def _collect_grad_samples(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['grad_samples'].append(p.grad)

    def _update(self):
        for group in self.param_groups:
            lr = group['lr']
            lamb = group['prior_precision']
            n = group['data_size']
            d = group['dampening']
            m = group['momentum_grad']
            h = group['momentum_hess']
            for p in group['params']:
                if p.requires_grad:
                    self._update_momentum_grad_buffers(p, lamb, n, m)
                    self._update_momentum_hess_buffers(p, lamb, n, d, h)
                    self._update_param_averages(p, lr)

    def _restore_param_averages(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.data = self.state[p]['param_average']
                    self.state[p]['param_average'] = None

    def _update_momentum_grad_buffers(self, p, lamb, n, m):
        m_grad = self.state[p]['momentum_grad_buffer']
        p_avg = self.state[p]['param_average']
        grad_avg = torch.mean(torch.stack(
            self.state[p]['grad_samples'], dim=0), dim=0)
        self.state[p]['momentum_grad_buffer'] = \
            m * m_grad + (1 - m) * ((lamb / n) * p_avg + grad_avg)

    def _update_momentum_hess_buffers(self, p, lamb, n, d, h):
        m_hess = self.state[p]['momentum_hess_buffer']
        p_avg = self.state[p]['param_average']
        temp = [(ps - p_avg) * g for ps, g in zip(
            self.state[p]['param_samples'],
            self.state[p]['grad_samples'])]
        # # won't work naively ...
        # new_mh = lamb / n + d + n * m_hess * torch.mean(
        #     torch.stack(temp, dim=0), dim=0)
        # ensure new hess has positive elements
        new_mh = torch.relu(lamb / n + n * m_hess * torch.mean(
            torch.stack(temp, dim=0), dim=0) + d)
        self.state[p]['momentum_hess_buffer'] = h * m_hess + (1 - h) * new_mh

    def _update_param_averages(self, p, lr):
        p_avg = self.state[p]['param_average']
        m_grad = self.state[p]['momentum_grad_buffer']
        m_hess = self.state[p]['momentum_hess_buffer']
        self.state[p]['param_average'] = p_avg - lr * m_grad / m_hess

    def _sample_weight(self):
        for group in self.param_groups:
            n = group['data_size']
            for p in group['params']:
                if p.requires_grad:
                    m_hess = self.state[p]['momentum_hess_buffer']
                    p_avg = self.state[p]['param_average']
                    normal_sample = torch.randn_like(p)
                    p_sample = normal_sample * torch.rsqrt(n * m_hess) + p_avg
                    p.data = p_sample

    @contextmanager
    def sampled_params(self):
        self._stash_param_averages()
        self._sample_weight()

        yield

        self._restore_param_averages()


class IVON(VON):

    def _update(self):
        for group in self.param_groups:
            lr = group['lr']
            lamb = group['prior_precision']
            n = group['data_size']
            d = group['dampening']
            m = group['momentum_grad']
            h = group['momentum_hess']
            for p in group['params']:
                if p.requires_grad:
                    self._update_momentum_grad_buffers(p, lamb, n, m)
                    self._update_param_averages(p, lr)
                    self._update_momentum_hess_buffers(p, lamb, n, d, h)

    def _update_momentum_hess_buffers(self, p, lamb, n, d, h):
        m_hess = self.state[p]['momentum_hess_buffer']
        p_avg = self.state[p]['param_average']
        temp = [(ps - p_avg) * g for ps, g in zip(
            self.state[p]['param_samples'],
            self.state[p]['grad_samples'])]
        gs = lamb / n + d - m_hess + n * m_hess * torch.mean(
            torch.stack(temp, dim=0), dim=0)
        self.state[p]['momentum_hess_buffer'] = \
            m_hess + (1 - h) * gs + 0.5 * ((1 - h) ** 2) * (gs ** 2) / m_hess
