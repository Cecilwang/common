import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()


def ground_truth(u, A, lam):
    w = cp.Variable(u.shape[0])
    cp.Problem(cp.Minimize(cp.quad_form((w - u), A) +
                           lam * cp.norm1(w))).solve()
    return np.array(w.value)


def draw(ws, label):
    w_hat = ws[-1]
    dist = [np.abs(w_hat - w).sum() for w in ws]
    ax = fig.get_axes()[0]
    ax.plot([i for i in range(ws.shape[0])], dist, label=label)
    ax = fig.get_axes()[1]
    ax.semilogy([i for i in range(ws.shape[0])], dist, label=label)


def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.)


def proximal_gradient_descent(u, A, lam):
    lr = 1.0 / np.linalg.eigh(2 * A)[0].max()
    w = np.zeros(u.shape)
    ws = np.zeros((100, *w.shape))
    for i in range(100):
        w = soft_thresholding(w - lr * 2 * (w - u) @ A, lr * lam)
        ws[i] = w
        loss = (w - u) @ A @ (w - u) + np.linalg.norm(w, 1)
    draw(ws, str(lam))
    return w


def main():
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.title.set_text("dist")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.title.set_text("semi-log dist")

    u = np.array([1.0, 2.0])
    A = np.array([[3.0, 0.5], [0.5, 1.0]])
    lam = 2.0
    for lam in [0.0, 2.0, 4.0, 6.0]:
        np.testing.assert_allclose(ground_truth(u, A, lam),
                                   proximal_gradient_descent(u, A, lam), 1e-6,
                                   1e-6)

    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    main()
