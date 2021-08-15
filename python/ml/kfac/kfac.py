import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.util import logger

log = logger("KFAC")


def linear_forward_hook(m, input, output):
    A = torch.einsum("bi,bj->bij", input[0], input[0]).mean(0)
    if hasattr(m.weight, "A"):
        m.weight.A = 0.95 * m.weight.A + 0.05 * A
    else:
        m.weight.A = A


def linear_backward_hook(m, g_input, g_output):
    d = g_output[0]
    G = torch.einsum('bi,bj->ij', d, d)
    if hasattr(m.weight, "G"):
        m.weight.G = 0.95 * m.weight.G + 0.05 * G
    else:
        m.weight.G = G


def classification_sampling(model_output):
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

    p = F.softmax(model_output, -1)  # BxC
    samples = torch.multinomial(p, 1, replacement=True)  # Bx1
    samples = samples.squeeze()  # B
    return samples


Hooks = {
    nn.Linear: (linear_forward_hook, linear_backward_hook),
}


class KFAC(torch.optim.Optimizer):

    def __init__(self,
                 parameters,
                 lr=0.01,
                 damping=1.0,
                 cov_intvl=10,
                 inv_intvl=100):
        super().__init__(parameters, dict())
        self.lr = lr
        self.damping = damping
        self.cov_intvl = cov_intvl
        self.inv_intvl = inv_intvl
        self.steps = 0
        self.hook_on = False

    def hook_wrapper(self, impl):

        def _wrapper(m, input, output):
            if self.hook_on:
                impl(m, input, output)

        return _wrapper

    def register(self, module):
        if isinstance(module, nn.Module):
            hook = Hooks.get(type(module), None)
            if hook:
                module.register_forward_hook(self.hook_wrapper(hook[0]))
                module.register_full_backward_hook(self.hook_wrapper(hook[1]))
                log.info("{} registered hook".format(module))
            else:
                log.warning("Skipped to register {}".format(type(module)))
            for x in module.modules():
                if (x != module):
                    self.register(x)

    def calc_inverse(self, p):
        if hasattr(p, "A") and hasattr(p, "G"):
            NG = p.G.shape[0]
            NA = p.A.shape[0]
            pi = torch.trace(p.A) / torch.trace(p.G) * NG / NA
            regG = torch.eye(NG) * torch.sqrt(self.damping / pi)
            regA = torch.eye(NA) * torch.sqrt(self.damping * pi)
            p.invG = (p.G + regG).inverse()
            p.invA = (p.A + regA).inverse()

    def calc_natural_gradient(self, p):
        if hasattr(p, "invG") and hasattr(p, "invA"):
            return p.invG @ p.grad @ p.invA
        else:
            return p.grad

    def step(self):
        for pg in self.param_groups:
            for p in pg["params"]:
                if self.steps % self.inv_intvl == 0:
                    self.calc_inverse(p)
                p.data -= self.lr * self.calc_natural_gradient(p)
        self.steps += 1


class EKFAC(KFAC):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def calc_inverse(self, p):
        pass

    def calc_natural_gradient(self, p):
        if hasattr(p, "A") and hasattr(p, "G"):
            Dg, Pg = torch.linalg.eigh(p.G)
            Da, Pa = torch.linalg.eigh(p.A)
            V = Pg.T @ p.grad @ Pa
            V = V / (torch.outer(Dg, Da) + self.damping)
            return Pg @ V @ Pa.T
        else:
            return p.grad
