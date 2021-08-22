class Metric(object):

    def __init__(self):
        self.name = "none"
        self.n = 0
        self.val = 0

    def reset(self):
        self.n = 0
        self.val = 0

    def calc(self, output, target):
        raise NotImplementedError

    def __iadd__(self, update):
        n, val = self.calc(update[0], update[1])
        self.n += n
        self.val += val
        return self

    def __call__(self):
        return self.val / self.n

    def __str__(self):
        return "{}: {}".format(self.name, self.val / self.n)


class Progress(Metric):

    def __init__(self, n):
        Metric.__init__(self)
        self.name = "Progress"
        self.n = n

    def reset(self):
        self.val = 0

    def calc(self, output, target):
        return 0, 1

    def __iadd__(self, update):
        super().__iadd__(update)
        if self.val > self.n:
            self.val %= self.n

    def __str__(self):
        width = len(str(self.n))
        return "{{}}:{{:>{0}}}/{{}}({{:3.0f}}%)".format(width).format(
            self.name, self.val, self.n, 100. * self.val / self.n)


class Loss(Metric):

    def __init__(self, loss_fn):
        Metric.__init__(self)
        self.name = "Loss"
        self.loss_fn = loss_fn

    def calc(self, output, target):
        return len(output), self.loss_fn(output, target).item() * len(output)

    def __str__(self):
        return "{}: {:.5f}".format(self.name, self.val / self.n)


class Accuracy(Metric):

    def __init__(self):
        Metric.__init__(self)
        self.name = "Accuracy"

    def calc(self, output, target):
        return len(output), output.argmax(dim=1).eq(target).sum().item()

    def __str__(self):
        return "{}: {:.0f}%".format(self.name, 100. * self.val / self.n)


class Metrics(object):

    def __init__(self, metrics):
        self.metrics = metrics

    def reset(self):
        for x in self.metrics:
            x.reset()

    def __iadd__(self, update):
        for x in self.metrics:
            x += update
        return self

    def __getitem__(self, i):
        return self.metrics[i]

    def __str__(self):
        return ", ".join([str(x) for x in self.metrics])
