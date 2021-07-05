import argparse
import jax
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
import timeit

fig = plt.figure()

def parse_args():
    parser = argparse.ArgumentParser(description='kfac.')
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--l', type=float, default=0.05)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--show', dest='show', action='store_true', default=False)
    return parser.parse_args()


def float2bcls(x):
    return 2 * (x > 0) - 1


def F(x, y, w):
    return -y * x @ w


def logloss(x, y, w):
    return jnp.log(1 + jnp.exp(F(x, y, w))).mean()


def l2reg(l, w):
    return l * w.T @ w


def loss(x, y, l, w):
    return (logloss(x, y, w) + l2reg(l, w)).reshape(())


def gen_data(args):
    n = args.n

    key = jax.random.PRNGKey(214)
    key, skey = jax.random.split(key)
    w = jax.random.normal(skey, shape=(2, 1))

    key, skey = jax.random.split(key)
    x = jax.random.normal(skey, shape=(n, 2))

    key, skey = jax.random.split(key)
    noise = args.noise * jax.random.normal(skey, shape=(n, 1))

    y = float2bcls(x @ w + noise)

    print("wstar^T: {} gets the loss: {}".format(w.T, loss(x, y, args.l, w)))

    if args.show:
        # Scatting data
        ax = fig.add_subplot(1, 2, 1)
        y = y.reshape(-1)
        ax.plot(x[y == 1, 0], x[y == 1, 1], "x")
        ax.plot(x[y == -1, 0], x[y == -1, 1], "o")
        y = y.reshape(-1, 1)
        ax.set_xlabel("x")
        ax.set_xlabel("y")

        # Ploting error surface
        w_min = w.min()
        w_max = w.max()
        W1 = jnp.arange(w_min-2, w_max+2, (4+w_max-w_min)/50.0)
        W1, W2 = jnp.meshgrid(W1, W1)
        l = jnp.array([
            loss(x, y, args.l, jnp.array([[w1], [w2]]))
            for w1, w2 in zip(W1.flatten(), W2.flatten())
        ]).reshape(W1.shape)

        what = jnp.array([[W1[jnp.unravel_index(l.argmin(), W1.shape)]],
                      [W2[jnp.unravel_index(l.argmin(), W2.shape)]]])
        print("what^T: {} gets the minimum loss: {}".format(what.T, l.min()))

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        surf = ax.plot_surface(W1, W2, l, alpha=0.6)
        ax.scatter(*what.flatten().tolist(), l.min(), color='red')

    return x, y

def kfac(args, x, y):
    f = lambda w: loss(x, y, args.l, w)

    key = jax.random.PRNGKey(412)
    w = jax.random.normal(key, shape=(2, 1))

    _h_fn = jax.hessian(f)
    h_fn = lambda w: _h_fn(w).reshape(w.shape[0], w.shape[0])

    for epoch in range(10):
        l, g_fn = jax.vjp(f, w)
        g = g_fn(jnp.ones(l.shape))[0]
        h = h_fn(w)

        w -= jnp.linalg.pinv(h) @ g

        if args.show:
            ax = fig.get_axes()[1]
            ax.scatter(*w.flatten().tolist(), l, color='blue', alpha=0.8)

        if epoch % 1 == 0:
            print("epoch: {:<3} loss: {}, gradient: {}, w: {}".format(
                epoch, l, g.T, w.T))


def main():
    args = parse_args()
    x, y = gen_data(args)
    kfac(args, x, y)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
