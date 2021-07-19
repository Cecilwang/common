import jax
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
jnp.set_printoptions(formatter=dict(float=lambda x: "{:+5.4f}".format(x)))


def F(x, w):
    return x @ w


def softmax(x):
    e = jnp.exp(x)
    s = e.sum(1, keepdims=True)
    return e / s


def cross_entropy(y, y_pred):
    return -jnp.log((softmax(y_pred) * y).sum(1)).mean()


def l2reg(l, w):
    return l * w.flatten().T @ w.flatten()


def loss(x, y, l, w):
    return (cross_entropy(y, F(x, w)) + l2reg(l, w)).reshape(())


def gen_data():
    key = jax.random.PRNGKey(214)

    n = 200

    key, skey = jax.random.split(key)
    x = 3 * (jax.random.normal(skey, shape=(n, 4)) - 0.5)

    key, skey = jax.random.split(key)
    noise = 0.5 * jax.random.normal(skey, shape=(n, 3))

    w = jnp.array([[2, -1, 0.5], [-3, 2, 1], [1, 2, 3]])

    y = jnp.hstack((x[:, 1:3], jnp.ones((n, 1)))) @ w
    y = jax.nn.one_hot(jnp.argmax(y, 1), 3)

    return x, y


def init_w():
    key = jax.random.PRNGKey(412)
    w = jax.random.normal(key, shape=(4, 3))
    return w


def update(x, y, lam, w, delta, epoch):
    w -= delta
    l = loss(x, y, lam, w)
    if epoch % 100 == 0:
        print("epoch: {:<3} loss: {:5.4f}, w: {}".format(epoch, l, w.flatten()))
    return w


def draw(ws, label):
    w_hat = ws[-1]
    dist = [np.abs(w_hat - w).sum() for w in ws]
    ax = fig.get_axes()[0]
    ax.plot([i for i in range(ws.shape[0])], dist, label=label)
    ax = fig.get_axes()[1]
    ax.semilogy([i for i in range(ws.shape[0])], dist, label=label)


def gradient_discent(x, y, lam, lr, epoch):
    w = init_w()
    ws = np.zeros((epoch, *w.shape))
    for e in range(epoch):
        g = jax.grad(loss, 3)(x, y, lam, w)
        delta = lr * g
        w = update(x, y, lam, w, delta, e)
        ws[e] = w
    draw(ws, "sgd")


def newton(x, y, lam, lr, epoch):
    w = init_w()
    ws = np.zeros((epoch, *w.shape))
    for e in range(epoch):
        g = jax.grad(loss, 3)(x, y, lam, w).reshape(12, 1)
        h = jax.hessian(loss, 3)(x, y, lam, w).reshape(12, 12)
        delta = lr * (jnp.linalg.pinv(h) @ g).reshape(4, 3)
        w = update(x, y, lam, w, delta, e)
        ws[e] = w
    draw(ws, "newton")


def main():
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.title.set_text("dist")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.title.set_text("semi-log dist")

    x, y = gen_data()

    lam, lr = 0.01, 0.1
    gradient_discent(x, y, lam, lr, 801)
    print("=" * 50)
    newton(x, y, lam, lr, 201)

    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()
