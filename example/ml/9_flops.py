from contextlib import contextmanager

import torch
import torch.nn as nn

batch = 10
n = 32
x = torch.ones([batch, n])


class ForwardHook:
    def __call__(self, module, inputs, outputs):
        module.foo = inputs[0] * inputs[0]


class BackwardHook:
    def __call__(self, module, grad_input, grad_output):
        module.bar = grad_output[0] * grad_output[0]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)

    def forward(self, x):
        x = x * x
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = Model()
model.fc1.register_forward_hook(ForwardHook())
model.fc1.register_full_backward_hook(BackwardHook())

#==============================================================================


def push_scope(name):
    tracing_state = torch._C._get_tracing_state()
    if tracing_state:
        tracing_state.push_scope(name)


def pop_scope():
    tracing_state = torch._C._get_tracing_state()
    if tracing_state:
        tracing_state.pop_scope()


class ScopePushHook:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        push_scope(self.name)


class ScopePopHook:
    def __call__(self, *args, **kwargs):
        pop_scope()


@contextmanager
def TraceScope(name):
    push_scope(name)
    yield
    pop_scope()


def trace_func(name, func):
    def _func(*args, **kwargs):
        with TraceScope(name):
            func(*args, **kwargs)

    return _func


def list_module(model, prefix):
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        #yield from list_module(module, submodule_prefix)
        yield from list_module(module, name)


class TraceWrapper(nn.Module):
    def __init__(self, model, name):
        super().__init__()
        self.model = model
        self.name = name
        for name, m in list_module(self.model, self.name):
            for k, hook in m._forward_hooks.items():
                if isinstance(hook, ForwardHook):
                    m._forward_hooks[k] = trace_func(hook.__class__.__name__,
                                                     hook)
            m.register_forward_pre_hook(ScopePushHook(name))
            m.register_forward_hook(ScopePopHook())

    def forward(self, *args, **kwargs):
        outputs = [self.model(*args, **kwargs)]
        for k, v in list_module(self.model, self.name):
            if k == 'fc1':
                outputs.append(v.foo)
        return outputs if len(outputs) > 1 else outputs[0]


#==============================================================================

model = TraceWrapper(model, 'model')
graph, _ = torch.jit._get_trace_graph(model, x)

for x in graph.nodes():
    if x.kind().startswith('aten'):
        print(f'{x.scopeName()} {x.kind()}')
        inputs = [y.type().sizes() for y in x.inputs() if y.isCompleteTensor()]
        print(f'inputs {inputs}')
        outputs = [
            y.type().sizes() for y in x.outputs() if y.isCompleteTensor()
        ]
        print(f'outputs {outputs}')
    print("================")
