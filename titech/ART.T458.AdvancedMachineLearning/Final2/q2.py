import torch


def main():
    w = torch.tensor([[1., 1.], [-1., -1.]], requires_grad=True)
    b = torch.tensor([-0.5, 1.5], requires_grad=True)
    q = torch.tensor([1., 1.], requires_grad=True)
    c = torch.tensor([-1.5], requires_grad=True)
    h = lambda x: torch.nn.ReLU()(w @ x + b)
    y = lambda x: torch.nn.Sigmoid()(h(x).T @ q + c)

    x = torch.tensor([0., 0.])
    print("h = {}".format(h(x)))
    print("y = {}".format(y(x)))

    x = torch.tensor([1., 1.])
    loss = -y(x) + torch.log(torch.exp(y(x)) + torch.exp(1.0 - y(x)))
    loss.backward()
    print("loss = {}".format(loss))
    print("gw = {}".format(w.grad))
    print("gb = {}".format(b.grad))
    print("gq = {}".format(q.grad))
    print("gc = {}".format(c.grad))


if __name__ == "__main__":
    main()
