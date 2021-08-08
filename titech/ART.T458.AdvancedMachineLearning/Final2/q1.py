import torch
import matplotlib.pyplot as plt
import numpy as np


def main():
    w = torch.tensor([[1000. for _ in range(6)]])
    v = torch.tensor([[2.], [-2.], [-1.], [1.], [3.], [-3.]])
    b = [[5., -995., -995., -1995.0, -1995.0, -2995.0]]
    b = torch.tensor(b, requires_grad=True)
    f = lambda x: torch.sigmoid(x @ w + b) @ v

    x = []
    y = []
    data = [[-1, 0, 0.], [0, 1, 2.], [1, 2, -1.], [2, 3, 3.], [3, 4, 0.]]
    for l, r, val in data:
        x += [i for i in np.arange(l, r, 0.1)]
        y += [val for i in range(int((r - l) / 0.1))]
    x = torch.tensor(x, dtype=torch.float32).reshape(len(x), 1)
    y = torch.tensor(y).reshape(len(y), 1)

    for i in range(1000000):
        loss = torch.norm(f(x) - y, p=1)
        loss.backward()
        if loss <= 1e-4:
            break
        with torch.no_grad():
            b -= 10 * b.grad
            b.grad.zero_()
    print(b.data, loss.item())

    x = [i / 10.0 for i in range(-10, 40, 1)]
    y = [f(torch.tensor([[i]])).item() for i in x]
    plt.plot(x, y)
    plt.show()

    x = [i for i in np.arange(-10, 10, 0.01)]
    y = torch.sigmoid(torch.tensor(x)).numpy()
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
