import argparse
from copy import copy
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cg', default='cg.in', type=str)

    subparsers = parser.add_subparsers()

    parser_manual = subparsers.add_parser('manual')
    parser_manual.set_defaults(method='manual')
    parser_manual.add_argument('--cached', nargs='+', default=[])
    parser_manual.add_argument('--verbose', type=str, default='True')
    parser_manual.add_argument('--show', type=str, default='True')

    parser_search = subparsers.add_parser('search')
    parser_search.set_defaults(method='search')
    parser_search.add_argument('--verbose', type=str, default='False')
    parser_search.add_argument('--show', type=str, default='False')
    return parser.parse_args()


class cprint(object):
    def __init__(self, color=None):
        self.color = {'red': '\033[91m'}.get(color, None)
        self.end = '\033[0m'

    def __call__(self, str, **kwargs):
        if self.color:
            print(self.color + str + self.end, **kwargs)
        else:
            print(str)


def setup_print(enable):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if enable or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class GIF:
    def __init__(self, cg, memory):
        self.cg = cg
        self.memory = memory
        self.setup_pos()
        self.Gs = []

    def setup_pos(self):
        self.pos = {}
        x = y = 0
        ops = ['op' + str(i) for i in range(1, len(self.cg.ops) - 1)]
        ops = ['input'] + ops + ['output']
        for i in range(len(ops)):
            self.pos[ops[i]] = (x, y)
            if i < ((len(ops)) // 2):
                x += 1
            else:
                x -= 1
            y -= 1

    def frame(self, title, processing=None):
        G = nx.Graph(title=title)
        for x in self.cg.ops.keys():
            G.add_node(x, c='green' if x == processing else 'blue')
        for y in self.cg.ops.values():
            for x in y.inputs:
                #if x in self.memory:
                #    if x.cached:
                #        color = 'orange'
                #    else:
                #        color = 'black'
                #else:
                #    if x.cached:
                #        color = 'yellow'
                #    else:
                #        color = 'gray'
                color = 'red' if x in self.memory else 'black'
                G.add_edge(x.op.name, y.name, c=color)
        self.Gs.append(G)

    def show(self):
        def draw_graph(i):
            G = self.Gs[i]
            node_color = [G.nodes[x]['c'] for x in G.nodes()]
            edge_color = [G[u][v]['c'] for u, v in G.edges()]
            plt.gca().set_title(G.graph['title'])
            nx.draw(G,
                    pos=self.pos,
                    node_color=node_color,
                    edge_color=edge_color,
                    with_labels=True)

        ani = animation.FuncAnimation(plt.figure(),
                                      draw_graph,
                                      frames=len(self.Gs),
                                      interval=1500,
                                      repeat=False)
        plt.show()


class Op:
    def __init__(self, name, inputs):
        self.name = name
        self.runtime = 1
        self.inputs = set(inputs)
        self._outputs = [Tensor(self, name + ':0')]
        self.outputs = set(self._outputs)
        self.children = set()
        # build edge
        for x in self.inputs:
            x.children |= self.outputs
            x.op.children.add(self)

    @property
    def size(self):
        return sum([x.size for x in self.inputs | self.outputs])

    def __str__(self):
        return f'({",".join([x.name for x in self.inputs])}) -> {self.name}'


class Tensor:
    def __init__(self, op, name, size=1):
        self.op = op
        self.name = name
        self.size = size
        self.children = set()
        self.cached = False

    def __str__(self):
        return f'{self.name} {self.size}'


class ComputationGraph:
    def __init__(self, filename):
        self.ops = {}
        self.tensors = []
        with open(filename) as f:
            for line in f:
                term = line.strip().split(' ')
                # Assume ops are defined in topological order
                op = Op(term[0], [self.get_tensor(x) for x in term[1:]])
                self.ops[op.name] = op
                self.tensors += op._outputs
        self['input'].runtime = 0
        self['input:0'].cached = True
        self['output'].runtime = 0

    def get_tensor(self, tensor):
        [op, index] = tensor.split(':')
        return self.ops[op]._outputs[int(index)]

    def is_tensor(self, name):
        return ':' in name

    def __getitem__(self, op_or_tensor):
        if self.is_tensor(op_or_tensor):
            return self.get_tensor(op_or_tensor)
        else:
            return self.ops[op_or_tensor]

    def __str__(self):
        return '\n'.join([str(x) for x in self.ops.values()])


class Memory:
    def __init__(self):
        self.memory = set()
        self.max_size = 0

    @property
    def size(self):
        return sum([x.size for x in self.memory])

    def _wrap_arg(self, arg):
        if isinstance(arg, Tensor):
            return set([arg])
        elif isinstance(arg, list):
            return set(arg)
        return arg

    def __and__(self, other):
        return self.memory & other

    def __ior__(self, other):
        self.memory |= self._wrap_arg(other)
        self.max_size = max(self.max_size, self.size)
        print(self)
        return self

    def __isub__(self, other):
        self.memory -= self._wrap_arg(other)
        print(self)
        return self

    def __iand__(self, other):
        self.memory &= self._wrap_arg(other)
        print(self)
        return self

    def __contains__(self, tensor):
        return tensor in self.memory

    def __str__(self):
        return 'Memory: ' + ' '.join([x.name for x in self.memory])


def execute(cg, args):
    order = []
    memory = Memory()
    gif = GIF(cg, memory)

    # yapf: disable
    def dfs(tensor,    # required tensor
            order,     # current computation order
            memory):   # current memroy status
        # yapf: enable

        # Execute tensor.op to produce it
        op = tensor.op
        # Execute if all inputs in the memory
        while memory & op.inputs != op.inputs:
            for x in op.inputs:
                if x in memory:
                    continue
                # Require input which is not in the memory recursively
                dfs(x, order, memory)

        print('=' * 10)
        print(op.name)
        # Execute !
        order.append(op)
        # Produce tensor and put it in the memory
        memory |= tensor
        gif.frame(f'executing {op.name}', op.name)
        # Remove unnecessary uncached inputs
        memory -= [x for x in op.inputs if not x.cached]
        gif.frame(f'removing inputs of {op.name}', op.name)
        # Remove unrequired cached tensors
        cut = copy(memory.memory)
        q = [cg['output:0']]
        reach = set()
        while len(q) > 0:
            x = q.pop(0)
            reach.add(x)
            if x in cut:
                if not x.cached:
                    cut.discard(x)
                continue
            q += x.op.inputs
        reach.add(cg['input:0'])  # walk aroud input
        if (memory & reach) != memory.memory:
            memory &= reach
            gif.frame(f'removing unrequired tensors', op.name)

    dfs(cg['output:0'], order, memory)
    cprint('red')(f'order: {" ".join([x.name for x in order])}', force=True)
    runtime = sum([x.runtime for x in order])
    cprint('red')(f'runtime: {runtime}', force=True)
    cprint('red')(f'max_memory: {memory.max_size}', force=True)
    if args.show == 'True':
        gif.show()
    return memory.max_size, runtime


def manual(cg, args):
    for x in args.cached:
        cg[x].cached = True
    execute(cg, args)
    for x in args.cached:
        cg[x].cached = False


def search(cg, args):
    runtimes = {}

    def dfs(i, binary):
        if i == len(cg.tensors):
            cprint('red')(f'{binary}', force=True)
            memory, runtime = execute(cg, args)
            runtimes[memory] = min(runtimes.get(memory, sys.maxsize), runtime)
            return
        if cg.tensors[i].name != 'input:0':
            cg.tensors[i].cached = True
            dfs(i + 1, binary + ('1' if cg.tensors[i].cached else '0'))
            cg.tensors[i].cached = False
        dfs(i + 1, binary + ('1' if cg.tensors[i].cached else '0'))

    dfs(0, '')
    plt.bar(runtimes.keys(), runtimes.values())
    plt.xticks(range(min(runtimes.keys()), max(runtimes.keys()) + 1))
    plt.ylabel('runtime')
    plt.xlabel('memory')
    plt.show()


def main(args):
    cg = ComputationGraph(args.cg)
    setup_print(args.verbose == 'True')

    if args.method == 'manual':
        manual(cg, args)
    else:
        search(cg, args)


if __name__ == "__main__":
    main(parse_args())
