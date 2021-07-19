import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(formatter=dict(float=lambda x: "{:+5.4f}".format(x)))

np.random.seed(214)
fig = plt.figure()


def gen_data():
    n = 200
    x = 3 * (np.random.randn(n, 4) - 0.5)
    y = (2 * x[:, 1] - 1 * x[:, 2] + 0.5 + 0.5 * np.random.randn(n)) > 0
    y = 2 * y - 1
    y = y.reshape(n, 1)
    return x, y


def F(x, y, w):
    return -y * (x @ w)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logloss(x, y, w):
    return np.log(1 + np.exp(F(x, y, w))).mean()


def l2reg(lam, w):
    return lam * w.T @ w


def loss(x, y, lam, w):
    return logloss(x, y, w).reshape(()) + l2reg(lam, w).reshape(())


def draw(ws, label):
    w_hat = ws[-1]
    dist = [np.abs(w_hat - w).sum() for w in ws]
    ax = fig.get_axes()[0]
    ax.plot([i for i in range(ws.shape[0])], dist, label=label)
    ax = fig.get_axes()[1]
    ax.semilogy([i for i in range(ws.shape[0])], dist, label=label)


def vec2diag(x):
    return np.diag(x.flatten())


def gradient(x, y, lam, w):
    D = (1 - sigmoid(-F(x, y, w))) * -y
    D = vec2diag(D)
    return (x.T @ D).mean(1, keepdims=True) + 2 * lam * w


def gradient_discent(x, y, lam, w, lr, epoch):
    ws = np.ndarray((epoch, *w.shape))
    for e in range(epoch):
        l = loss(x, y, lam, w)
        g = gradient(x, y, lam, w)
        w -= lr * g
        ws[e] = w
        if e % 50 == 0:
            print("epoch: {:<3} loss: {:5.4f}, gradient: {}, w: {}".format(
                e, l, g.T, w.T))
    draw(ws, "sgd")


def hessian(x, y, lam, w):
    D = sigmoid(-F(x, y, w)) * (1 - sigmoid(-F(x, y, w)))  # ignore y^2 == 1
    D = vec2diag(D)
    H = x.T @ D @ x / x.shape[0]
    np.fill_diagonal(H, H.diagonal() + 2 * lam)
    return H


def newton(x, y, lam, w, lr, epoch):
    ws = np.ndarray((epoch, *w.shape))
    for e in range(epoch):
        l = loss(x, y, lam, w)
        g = gradient(x, y, lam, w)
        h = hessian(x, y, lam, w)
        w -= lr * np.linalg.pinv(h) @ g
        ws[e] = w
        if e % 50 == 0:
            print("epoch: {:<3} loss: {:5.4f}, gradient: {}, w: {}".format(
                e, l, g.T, w.T))
    draw(ws, "newton")


def main():
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.title.set_text("dist")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.title.set_text("semi-log dist")

    x, y = gen_data()
    w = np.random.randn(x.shape[-1], y.shape[-1])
    lam, lr = 0.01, 0.1
    gradient_discent(x, y, lam, w.copy(), lr, 351)
    print("=" * 50)
    newton(x, y, lam, w.copy(), lr, 101)

    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()
