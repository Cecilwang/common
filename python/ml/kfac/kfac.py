import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.util import logger

log = logger("KFAC")


class Config(object):

    def __init__(self):
        self.compatible_with_backpack = False


config = Config()


class Hook(object):

    def register(self, module):
        module.register_forward_hook(self.forward_hook())
        if config.compatible_with_backpack:
            module.register_backward_hook(self.backward_hook())
        else:
            module.register_full_backward_hook(self.backward_hook())
        log.info("{} registered {}".format(module, type(self)))

    def forward_hook(self):

        def _forward_hook(m, input, output):
            pass

        return _forward_hook

    def backward_hook(self):

        def _backward_hook(m, g_input, g_output):
            pass

        return _backward_hook


class ReLUHook(Hook):

    def forward_hook(self):

        def _forward_hook(m, input, output):
            m.input = input[0]

        return _forward_hook

    def backward_hook(self):

        def _backward_hook(m, g_input, g_output):
            m.d = m.succ.d * torch.gt(m.input, 0).float()

        return _backward_hook


class LinearHook(Hook):

    def forward_hook(self):

        def _forward_hook(m, input, output):
            A = torch.einsum("bi,bj->bij", input[0], input[0]).mean(0)
            if hasattr(m.weight, "A"):
                m.weight.A = 0.95 * m.weight.A + 0.05 * A
            else:
                m.weight.A = A

        return _forward_hook

    def backward_hook(self):

        def _backward_hook(m, g_input, g_output):
            d = m.succ.d
            G = torch.einsum('bi,bj->ij', d, d)
            if hasattr(m.weight, "G"):
                m.weight.G = 0.95 * m.weight.G + 0.05 * G
            else:
                m.weight.G = G
            m.d = d @ m.weight

        return _backward_hook


class CrossEntropyLossHook(Hook):

    def forward_hook(self):

        def _forward_hook(m, input, output):
            m.model_output = input[0]

        return _forward_hook

    def backward_hook(self):

        def _backward_hook(m, g_input, g_output):
            #                               exp(model_output[i])
            # probability = softmax =  -----------------------------
            #                          sigma_j(exp(model_output[j]))
            #
            #                            exp(model_output[target])
            # CrossEntropyLoss  = - log( ------------------------- )
            #                            sum(exp(model_output[j]))
            #
            #                   = - log(p_target)
            #
            #                   = - output[target] + log(sum(exp(model_output[j])))
            #
            # CrossEntropyLoss' = p_i - (1 if i==target else 0) = p - y

            # Using Monte Carlo to estimate the derivatives of the loss function
            # Sampling target following the probability of the model "P(y|x,model)"

            B, C, N = *m.model_output.shape, 1  # batch,#classes,#samples
            p = F.softmax(m.model_output, -1)  # BxC
            samples = torch.multinomial(p, N, replacement=True)  # BxN
            samples = F.one_hot(samples, C)  # BxNxC
            p = p.unsqueeze(1).repeat(1, N, 1)  # BxNxC
            d = p - samples
            d = d.reshape(-1, p.shape[-1])  # BNxC
            # Handle average in advance
            d /= np.sqrt(N)
            if m.reduction == "mean":
                d /= np.sqrt(B)
            m.d = d

        return _backward_hook


Hooks = {
    nn.Linear: LinearHook(),
    nn.CrossEntropyLoss: CrossEntropyLossHook(),
    nn.ReLU: ReLUHook()
}


class KFAC(torch.optim.Optimizer):

    def __init__(self, parameters, lr=0.01, damping=1.0):
        super().__init__(parameters, dict(lr=lr, damping=damping))

    def register(self, modules, pre=None):
        if isinstance(modules, nn.Module):
            hook = Hooks.get(type(modules), None)
            if hook:
                hook.register(modules)
                if pre:
                    pre.succ = modules
                pre = modules
            else:
                log.warning("Skipped to register {}".format(type(modules)))
            for x in modules.modules():
                if (x != modules):
                    pre = self.register(x, pre)
        else:
            for x in modules:
                pre = self.register(x, pre)
        return pre

    def step(self):
        for pg in self.param_groups:
            lr = pg["lr"]
            damping = pg["damping"]
            for p in pg["params"]:
                if hasattr(p, "A") and hasattr(p, "G"):
                    NG = p.G.shape[0]
                    NA = p.A.shape[0]
                    pi = torch.trace(p.A) / torch.trace(p.G) * NG / NA
                    regG = torch.eye(NG) * torch.sqrt(damping / pi)
                    regA = torch.eye(NA) * torch.sqrt(damping * pi)
                    invG = (p.G + regG).inverse()
                    invA = (p.A + regA).inverse()
                    p.data += -lr * (invG @ p.grad @ invA)
                else:
                    p.data += -lr * p.grad


class EKFAC(KFAC):

    def __init__(self, parameters, lr=0.01, damping=1.0):
        super().__init__(parameters, lr, damping)

    def step(self):
        for pg in self.param_groups:
            lr = pg["lr"]
            damping = pg["damping"]
            for p in pg["params"]:
                if hasattr(p, "A") and hasattr(p, "G"):
                    Dg, Pg = torch.linalg.eigh(p.G)
                    Da, Pa = torch.linalg.eigh(p.A)
                    V = Pg.T @ p.grad @ Pa
                    V = V / (torch.outer(Dg, Da) + damping)
                    V = Pg @ V @ Pa.T
                    p.data += -lr * V
                else:
                    p.data += -lr * p.grad
