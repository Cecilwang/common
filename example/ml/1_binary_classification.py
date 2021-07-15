# Refs: https://colab.research.google.com/drive/1tkKCaDuZMhKK2UIgYKLRSxJMcHpp5qCT#scrollTo=L1nwXMbu2v96

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(214)
fig = plt.figure()


def float2bcls(x):
    return 2 * (x > 0) - 1


def vec2diag(x):
    return np.diag(x.flatten())


def gen_data():
    lam = 0.01
    n = 40
    w = np.random.randn(2, 1)
    noise = 0.8 * np.random.randn(n, 1)
    x = np.random.randn(n, 2)
    y = float2bcls(x @ w + noise)
    # print(np.hstack((x, y)))
    print("wstar^T: {} gets the loss: {}".format(w.T, loss(x, y, lam, w)))

    ax = fig.add_subplot(1, 2, 1)
    y = y.reshape(-1)
    ax.plot(x[y == 1, 0], x[y == 1, 1], "x")
    ax.plot(x[y == -1, 0], x[y == -1, 1], "o")
    y = y.reshape(-1, 1)
    ax.set_xlabel("x")
    ax.set_xlabel("y")
    return x, y, lam


def F(x, y, w):
    return -y * x @ w


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logloss(x, y, w):
    gt0, n = 0.0, x.shape[0]
    for (x1, x2), yi in zip(x, y):
        gt0 += np.log(1 + np.exp(-yi * (x1 * w[0][0] + x2 * w[1][0]))) / n
    gt1 = -np.log(sigmoid(-F(x, y, w))).mean()
    gt2 = np.log(1 + np.exp(F(x, y, w))).mean()
    np.testing.assert_almost_equal(gt0, gt1)
    np.testing.assert_almost_equal(gt0, gt2)
    return gt2


def l2reg(lam, w):
    return lam * w.T @ w


def loss(x, y, lam, w):
    return logloss(x, y, w) + l2reg(lam, w)


def gradient(x, y, lam, w):
    D = (1 - sigmoid(-F(x, y, w))) * -y
    D = vec2diag(D)
    return (x.T @ D).mean(1, keepdims=True) + 2 * lam * w


def surface(x, y, lam):
    k = 1.5
    W1 = np.arange(-k, 0, 0.02)
    W1, W2 = np.meshgrid(W1, W1)
    l = np.array([
        loss(x, y, lam, np.array([[w1], [w2]]))
        for w1, w2 in zip(W1.flatten(), W2.flatten())
    ]).reshape(W1.shape)

    w = np.array([[W1[np.unravel_index(l.argmin(), W1.shape)]],
                  [W2[np.unravel_index(l.argmin(), W2.shape)]]])
    np.testing.assert_almost_equal(l.min(), loss(x, y, lam, w))
    print("what^T: {} gets the minimum loss: {}".format(w.T, l.min()))

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    #surf = ax.plot_surface(W1, W2, l, cmap=cm.coolwarm)
    #fig.colorbar(surf)
    surf = ax.plot_surface(W1, W2, l, alpha=0.6)
    ax.scatter(*w.flatten().tolist(), l.min(), color="red")

    return ax


def gradient_discent(x, y, lam, w, ax):
    for epoch in range(50):
        l = loss(x, y, lam, w)
        ax.scatter(*w.flatten().tolist(), l, color="yellow", alpha=0.8)
        g = gradient(x, y, lam, w)
        w -= g
        if epoch % 10 == 0:
            print("epoch: {:<3} loss: {}, gradient: {}, w: {}".format(
                epoch, l, g.T, w.T))


def hessian(x, y, lam, w):
    D = sigmoid(-F(x, y, w)) * (1 - sigmoid(-F(x, y, w)))  # ignore y^2 == 1
    D = vec2diag(D)
    H = x.T @ D @ x / x.shape[0] + 2 * lam
    np.fill_diagonal(H, H.diagonal() + 2 * lam)
    return H


def newton(x, y, lam, w, ax):
    for epoch in range(50):
        l = loss(x, y, lam, w)
        ax.scatter(*w.flatten().tolist(), l, color="green", alpha=0.8)
        g = gradient(x, y, lam, w)
        h = hessian(x, y, lam, w)
        w -= np.linalg.pinv(h) @ g
        if epoch % 10 == 0:
            print("epoch: {:<3} loss: {}, gradient: {}, w: {}".format(
                epoch, l, g.T, w.T))


def main():
    x, y, lam = gen_data()
    ax = surface(x, y, lam)
    w = np.random.randn(2, 1)
    gradient_discent(x, y, lam, w.copy(), ax)
    newton(x, y, lam, w.copy(), ax)
    plt.show()


if __name__ == "__main__":
    main()
