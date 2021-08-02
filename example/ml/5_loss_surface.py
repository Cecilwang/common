import argparse
import copy
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch

from python import util

import model as M
from problem import MNIST

Color = cycle("bgrcmk")


def parse_args():
    parser = argparse.ArgumentParser(description="loss_surface")
    parser.add_argument("sfc", type=str)
    parser.add_argument("traj", type=str)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data", type=str, default="./.data")
    parser.add_argument("--log", type=str, default="./.log")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--step", type=int, default=5)
    return parser.parse_args()


def random_color(i, n, name='hsv'):
    #return plt.cm.get_cmap(name, n)(i)
    return next(Color)


def flatten_parameters(params):
    return torch.cat([
        p.view(p.numel()) if p.dim() > 1 else torch.FloatTensor(p)
        for p in params
    ])


def direction2parameters(direction, params_like):
    params = copy.deepcopy(params_like)
    i = 0
    for p in params:
        p.copy_(torch.tensor(direction[i:i + p.numel()]).view(p.size()))
        i += p.numel()
    assert i == len(direction)
    return params


def angle(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_directions(final_model, traj_files):
    p1 = [p.data for p in final_model.parameters()]

    matrix = []
    curr_model = copy.deepcopy(final_model)
    for model_files in traj_files:
        for f in model_files:
            M.load(curr_model, f)
            p2 = [p.data for p in curr_model.parameters()]
            d = [y - x for x, y in zip(p1, p2)]
            matrix.append(flatten_parameters(d).numpy())

    pca = PCA(n_components=2)
    pca.fit(np.array(matrix))
    d1 = np.array(pca.components_[0])
    d2 = np.array(pca.components_[1])
    print("Angle between directions: {}".format(angle(d1, d2)))
    return d1, d2


def project(final_model, traj_files, d1, d2, loss_fn):
    p1 = [p.data for p in final_model.parameters()]

    xs, ys, losses = [], [], []
    curr_model = copy.deepcopy(final_model)
    for model_files in traj_files:
        _xs, _ys, _losses = [], [], []
        for f in model_files:
            M.load(curr_model, f)
            p2 = [p.data for p in curr_model.parameters()]
            d = [y - x for x, y in zip(p1, p2)]
            d = flatten_parameters(d).numpy()

            x = np.dot(d, d1) / np.linalg.norm(d1)
            y = np.dot(d, d2) / np.linalg.norm(d2)
            loss = loss_fn(curr_model)
            d = np.linalg.norm(d1 * x + d2 * y - d)

            print("{} at({}, {}) {} {}".format(f, x, y, d, loss))

            _xs.append(int(x))
            _ys.append(int(y))
            _losses.append(loss)
        xs.append(_xs)
        ys.append(_ys)
        losses.append(_losses)

    return np.array(xs), np.array(ys), np.array(losses)


def get_surface(final_model, d1, d2, xs, ys, loss_fn, step):
    x_max = int(np.max([np.max(xs), 0]) + 1)
    x_min = int(np.min([np.min(xs), 0]))
    y_max = int(np.max([np.max(ys), 0]) + 1)
    y_min = int(np.min([np.min(ys), 0]))
    print("({}, {}), ({}, {})".format(x_min, y_min, x_max, y_max))

    x_stride = (x_max - x_min) // step
    y_stride = (y_max - y_min) // step

    X = [i for i in range(x_min, x_max, x_stride)]
    Y = [i for i in range(y_min, y_max, y_stride)]

    curr_model = copy.deepcopy(final_model)
    p = [p.data for p in curr_model.parameters()]
    final_point = flatten_parameters(p).numpy()
    loss = []
    for i in X:
        column = []
        for j in Y:
            p = direction2parameters(d1 * i + d2 * j + final_point, p)
            for x, y in zip(curr_model.parameters(), p):
                x.data = y
            column.append(loss_fn(curr_model))
        loss.append(column)
    loss = np.array(loss)

    Y, X = np.meshgrid(Y, X)
    return X, Y, loss


def get_traj_files(args):
    files = []
    for x in args.traj.split(","):
        traj = []
        for i in range(args.e + 1):
            traj.append("{}/{}-{}.pkl".format(args.log, x, i))
        files.append(traj)
    return files


def wrap_loss_fn(loss_fn, loader, device):

    def _loss(model):
        loss = 0.0
        for i, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss += loss_fn(output, target).item()
        return loss / i + 1

    return _loss


def draw(surface, trajectories):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title("Loss Surface")
    ax.plot_surface(*surface, alpha=0.8)

    traj = args.traj.split(",")
    for i, x in enumerate(traj):
        ax.plot3D(trajectories[0][i],
                  trajectories[1][i],
                  trajectories[2][i],
                  label=x,
                  color=random_color(i, len(traj)))
    ax.legend()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("loss")

    plt.show()


def main(args):
    final_model, loader, _, _loss_fn = MNIST(**vars(args))
    final_model.eval()
    M.load(final_model, "{}/{}-{}.pkl".format(args.log, args.sfc, args.e))
    loss_fn = wrap_loss_fn(_loss_fn, loader, args.device)

    traj_files = get_traj_files(args)
    d1, d2 = get_directions(final_model, traj_files)
    trajectories = project(final_model, traj_files, d1, d2, loss_fn)
    surface = get_surface(final_model, d1, d2, xs, ys, loss_fn, args.step)

    draw(surface, trajectories)


if __name__ == "__main__":
    main(parse_args())
