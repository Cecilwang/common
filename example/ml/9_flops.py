from math import prod

import torch
import torch.nn as nn
import torchvision


class FLOPsNode:
    def __init__(self, prefix, shapes={}):
        self.prefix = prefix
        self.name = prefix.split('.')[-1]
        self._parent = None
        self.children = {}
        # TODO: Assumption: DataType is FP32
        self.data_type_bytes = 4
        self.shapes = shapes
        for k, v in self.shapes.items():
            setattr(self, k, v)

    def __str__(self):
        shape_info = ' '.join([f'{k} {v}' for k, v in self.shapes.items()])
        shape_info = f'\033[94m{shape_info}\033[0m'
        flops_memory = {}
        for k in dir(self):
            if k.endswith('flops'):
                flops_memory[k] = f'{getattr(self, k):.2e}'
            if k.endswith('memory'):
                unit = ['B', 'KB', 'MB', 'GB']
                i = 0
                memory = getattr(self, k) / self.data_type_bytes
                while i < len(unit) - 1 and memory > 1024:
                    memory /= 1024
                    i += 1
                flops_memory[k] = f'{memory:.2f} {unit[i]}'

        profile_info = ' '.join([f'{k} {v}' for k, v in flops_memory.items()])
        profile_info = f'\033[91m{profile_info}\033[0m'
        return f'{self.name} {shape_info} {profile_info}'

    def to_string(self, dfs=True):
        if not dfs:
            return str(self)
        info = []
        indent = ''

        def dfs(x, indent=''):
            info.append(indent + str(x))
            indent += '  '
            for _, y in x.children.items():
                dfs(y, indent)

        dfs(self)
        return '\n'.join(info)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        self.prefix = f'{parent.prefix}.{self.name}'

    def __getitem__(self, path):
        path = path.split('.')
        node = self
        for x in path:
            if x == '':
                return node
            elif x in node.children:
                node = node.children[x]
            else:
                return getattr(node, x, None)
        return node

    def insert(self, path, node=None):
        if node is None:
            node = FLOPsNode(path)
        node.parent = self
        self.children[path] = node
        setattr(self, path, node)

    def __imul__(self, batch):
        for k in dir(self):
            if k.endswith('flops') or k.endswith('memory'):
                setattr(self, k, getattr(self, k) * batch)
        return self


def get_input_shape(node, i):
    return [x for x in node.inputs()][i].type().sizes()


def get_output_shape(node, i):
    return [x for x in node.outputs()][i].type().sizes()


class Cov(FLOPsNode):
    def __init__(self, name, inp):
        super().__init__(name)
        self.flops = prod(inp) * inp[-1]
        self.memory = inp[-1]**2


class Inv(FLOPsNode):
    def __init__(self, name, n):
        super().__init__(name)
        self.flops = n**3
        self.memory = n**2


class AorB(FLOPsNode):
    def __init__(self, name, inp):
        super().__init__(name, {'input': inp, 'output': [inp[-1], inp[-1]]})
        self.insert('cov', Cov(f'{name}.cov', inp))
        self.insert('inv', Inv(f'{name}.inv', inp[-1]))


class KFACPrecondition(FLOPsNode):
    def __init__(self, name, A, B):
        super().__init__(name)
        self.flops = A * A * B + B * B * A


class KFAC(FLOPsNode):
    def __init__(self, name, inp, grad):
        super().__init__(name)
        self.insert('A', AorB(f'{name}.A', inp))
        self.insert('B', AorB(f'{name}.B', grad))
        self.insert('pre', KFACPrecondition(f'{name}.pre', inp[-1], grad[-1]))


class Precondition(FLOPsNode):
    def __init__(self, name, grad):
        super().__init__(name)
        self.flops = grad**2


class LayerWise(FLOPsNode):
    def __init__(self, name, grad):
        super().__init__(name, {
            'gradient': grad,
            'output': [prod(grad[1:]), prod(grad[1:])]
        })
        self.insert('cov', Cov(f'{name}.cov', grad))
        self.insert('inv', Inv(f'{name}.inv', prod(grad[1:])))
        self.insert('pre', Precondition(f'{name}.pre', prod(grad[1:])))


class UnitWise(FLOPsNode):
    def __init__(self, name, inp, grad):
        super().__init__(name, {'input': inp, 'gradient': grad})
        cov = FLOPsNode(f'{name}.cov')
        cov.flops = inp[0] * prod(inp[1:])**2
        cov.flops += prod(grad)
        cov.flops += prod(grad) * prod(inp[1:])**2
        cov.memory = prod(grad[1:]) * prod(inp[1:])**2
        self.insert('cov', cov)
        inv = Inv(f'{name}.inv', prod(inp[1:]))
        inv *= prod(grad[1:])
        self.insert('inv', inv)
        pre = Precondition(f'{name}.pre', prod(inp[1:]))
        pre *= prod(grad[1:])
        self.insert('pre', pre)


class Linear(FLOPsNode):
    def __init__(self, name, node):
        super().__init__(
            name, {
                'input': get_input_shape(node, 0),
                'weight': get_input_shape(node, 1),
                'output': get_output_shape(node, 0)
            })
        self.flops = prod(self.input) * self.weight[-1]
        self.insert('kfac', KFAC(f'{name}.kfac', self.input, self.output))
        self.insert(
            'lw', LayerWise(f'{name}.lw',
                            [self.input[0], prod(self.weight)]))
        self.insert('uw', UnitWise(f'{name}.uw', self.input, self.output))

    @classmethod
    def kind(self):
        return 'aten::linear'

    @classmethod
    def type(self):
        return nn.Linear


class Conv2d(FLOPsNode):
    def __init__(self, name, node):
        super().__init__(
            name, {
                'input': get_input_shape(node, 0),
                'weight': get_input_shape(node, 1),
                'output': get_output_shape(node, 0)
            })
        self.flops = self.input[0] * prod(self.weight) * prod(self.output[2:])
        self.img2col_input = [
            self.input[0] * prod(self.output[2:]),  # batch*output
            self.input[1] * prod(self.weight[2:])  # in_channel*kernel
        ]
        self.img2col_output = [
            self.input[0] * prod(self.output[2:]),  # batch*output
            self.output[1]  # out_channel
        ]
        self.shapes['img2col_input'] = self.img2col_input
        self.shapes['img2col_output'] = self.img2col_output
        kfac = KFAC(f'{name}.kfac', self.img2col_input, self.img2col_output)
        self.insert('kfac', kfac)
        lw = LayerWise(f'{name}.lw', [self.input[0], prod(self.weight)])
        self.insert('lw', lw)
        uw = UnitWise(f'{name}.uw', self.img2col_input, self.img2col_output)
        self.insert('uw', uw)

    @classmethod
    def kind(self):
        return 'aten::conv2d'

    @classmethod
    def type(self):
        return nn.Conv2d


CLS = {}
for k, v in globals().copy().items():
    try:
        if issubclass(v, FLOPsNode):
            if hasattr(v, 'type') and hasattr(v, 'kind'):
                CLS[v.type()] = v
    except:
        pass


class FLOPs:
    def __init__(self):
        self.root = FLOPsNode('root')

    def insert(self, node):
        path = node.prefix.split('.')
        x = self.root
        for y in path[:-1]:
            if y not in x.children:
                x.insert(y)
            x = x.children[y]
        y = path[-1]
        assert y not in x.children
        x.insert(y, node)
        for k, v in self.root.children.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def __getitem__(self, path):
        return self.root[path]

    def get(self, path, default=None):
        ret = self[path]
        return default if ret is None else ret

    def __str__(self):
        info = []
        indent = ''

        def dfs(x, indent=''):
            if x != self.root:
                info.append(indent + str(x))
                indent += '  '
            for _, y in x.children.items():
                dfs(y, indent)

        dfs(self.root)
        return '\n'.join(info)


#TODO: support filter
def list_module(model, prefix):
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from list_module(module, submodule_prefix)


def profile(model, input_shape):
    graph = torch.jit.freeze(torch.jit.script(model.eval())).graph
    if isinstance(input_shape, list):
        input_shape = {1: input_shape}
    inps = list(graph.inputs())
    for k, v in input_shape.items():
        inps[k].setType(inps[k].type().with_sizes(v))
    torch._C._jit_pass_propagate_shapes_on_graph(graph)

    nodes = {k: graph.findAllNodes(v.kind()) for k, v in CLS.items()}
    flops = FLOPs()
    for name, m in list_module(model, ''):
        kind = type(m)
        if kind not in CLS:
            continue
        if len(nodes[kind]) < 1:
            print(f'cannot find {name} in shape graph')
            continue
        flops.insert(CLS[kind](name, nodes[kind][0]))
        nodes[kind].pop(0)
    return flops


def main():
    model = torchvision.models.resnet18()
    flops = profile(model, [32, 3, 224, 224])
    print(flops)
    print(flops['layer1.0.conv1.kfac.pre'])
    print(flops.fc.to_string(True))


if __name__ == "__main__":
    main()
