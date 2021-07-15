import argparse
import jax
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
import timeit

fig = plt.figure()
jnp.set_printoptions(formatter=dict(float=lambda x: "{:+10.8f}".format(x)))


def parse_args():
    parser = argparse.ArgumentParser(description="ggn")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--e", type=int, default=20)
    parser.add_argument("--de", type=int, default=5)
    parser.add_argument("--l", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--show",
                        dest="show",
                        action="store_true",
                        default=False)
    parser.add_argument("--w1_min", type=float, default=None)
    parser.add_argument("--w1_max", type=float, default=None)
    parser.add_argument("--w2_min", type=float, default=None)
    parser.add_argument("--w2_max", type=float, default=None)
    return parser.parse_args()


def float2bcls(x):
    return 2 * (x > 0) - 1


def F(x, w):
    return x @ w


def logloss(y, y_pred):
    return jnp.log(1 + jnp.exp(-y * y_pred)).mean()


def l2reg(l, w):
    return l * w.T @ w


def loss(x, y, l, w):
    return (logloss(y, F(x, w)) + l2reg(l, w)).reshape(())


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
        ax.plot(x[y > 0, 0], x[y > 0, 1], "x")
        ax.plot(x[y <= 0, 0], x[y <= 0, 1], "o")
        y = y.reshape(-1, 1)
        ax.set_xlabel("x")
        ax.set_xlabel("y")

        # Ploting error surface
        w1_min = w[0, 0] - 1.0 if args.w1_min == None else args.w1_min
        w1_max = w[0, 0] + 1.0 if args.w1_max == None else args.w1_max
        w2_min = w[1, 0] - 1.0 if args.w2_min == None else args.w2_min
        w2_max = w[1, 0] + 1.0 if args.w2_max == None else args.w2_max
        W1 = jnp.arange(w1_min, w1_max, (w1_max - w1_min) / 50.0)
        W2 = jnp.arange(w2_min, w2_max, (w2_max - w2_min) / 50.0)
        W1, W2 = jnp.meshgrid(W1, W2)
        l = jnp.array([
            loss(x, y, args.l, jnp.array([[w1], [w2]]))
            for w1, w2 in zip(W1.flatten(), W2.flatten())
        ]).reshape(W1.shape)

        what = jnp.array([[W1[jnp.unravel_index(l.argmin(), W1.shape)]],
                          [W2[jnp.unravel_index(l.argmin(), W2.shape)]]])
        print("what^T: {} gets the minimum loss: {}".format(what.T, l.min()))

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        surf = ax.plot_surface(W1, W2, l, alpha=0.6)
        ax.scatter(*what.flatten().tolist(), l.min(), color="gray")

    return x, y


def init_w():
    key = jax.random.PRNGKey(412)
    w = jax.random.normal(key, shape=(2, 1))
    return w


def update(x, y, w, delta, epoch, args, color):
    w -= delta
    l = loss(x, y, args.l, w)
    if args.show:
        ax = fig.get_axes()[1]
        ax.scatter(*w.flatten().tolist(), l, color=color, alpha=0.8)
    if epoch % args.de == 0 or epoch == args.e - 1:
        print("epoch: {:<3} loss: {:<20}, delta: {}, w: {}".format(
            epoch, l, delta.T, w.T))
    return w


def gradient_discent(args, x, y):
    w = init_w()
    for epoch in range(args.e):
        g = jax.grad(loss, 3)(x, y, args.l, w)
        # Choose the learning rate carefully, otherwise it will not converge
        delta = args.lr * g
        w = update(x, y, w, delta, epoch, args, "red")


def newton(args, x, y):
    w = init_w()
    for epoch in range(args.e):
        g = jax.grad(loss, 3)(x, y, args.l, w)
        h = jax.hessian(loss, 3)(x, y, args.l,
                                 w).reshape(w.shape[0], w.shape[0])

        # TODO(sxwang): It is not necessary to explicitly caculate the inverse
        # of Hessian. Because we can get inv(h)@g directly by solving this
        # linear question: Ax=b <=> h @ (inv(h)@g) = g, there are lots efficient
        # algorithms(e.g. CG) are faster than caculating inverse. And the
        # representation(callable function) of Ax can be easily derived by
        # hvp = jvp(vjp)

        # Since Newton's method can directly converge, omiting the learning rate
        delta = jnp.linalg.pinv(h) @ g
        w = update(x, y, w, delta, epoch, args, "green")


def ggn(args, x, y):
    w = init_w()
    for epoch in range(args.e):
        g = jax.grad(loss, 3)(x, y, args.l, w)

        G = jnp.zeros((w.shape[0], w.shape[0]))
        # TODO(sxwang): Optimizing this loop by matrix-matrix product
        for xi, yi in zip(x, y):
            xi = jnp.array([xi])
            yi = jnp.array([yi])
            dFdw = jax.grad(lambda w: F(xi, w).reshape(()))(w)
            hz = jax.hessian(logloss, 1)(yi, F(xi, w)).reshape(
                (w.shape[1], w.shape[1]))
            G += dFdw @ hz @ dFdw.T
        G = G / x.shape[0] + args.l * jnp.identity(G.shape[0])

        # The learning rate also matters here.
        delta = args.lr * jnp.linalg.pinv(G) @ g
        w = update(x, y, w, delta, epoch, args, "blue")


def main():
    args = parse_args()
    x, y = gen_data(args)
    gradient_discent(args, x, y)
    print("=" * 50)
    newton(args, x, y)
    print("=" * 50)
    ggn(args, x, y)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
