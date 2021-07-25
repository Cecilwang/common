import argparse
import copy
from itertools import cycle

import loss_landscapes
from loss_landscapes.model_interface.model_wrapper import wrap_model
from loss_landscapes.model_interface.model_parameters import rand_u_like
from loss_landscapes.model_interface.model_parameters import orthogonal_to
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from python import util

import model as M
from problem import MNIST

Color = cycle("bgrcmk")


def parse_args():
    parser = argparse.ArgumentParser(description="loss_surface")
    parser.add_argument("sfc", type=str)
    parser.add_argument("traj", type=str)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--data", type=str, default="./.data")
    parser.add_argument("--log", type=str, default="./.log")
    parser.add_argument("--dist", type=float, default=10)
    parser.add_argument("--step", type=int, default=50)
    return parser.parse_args()


def random_color(i, n, name='hsv'):
    #return plt.cm.get_cmap(name, n)(i)
    return next(Color)


def sfcfile(dir, sfc, i, j):
    return "{}/{}-{}-{}.pkl".format(dir, sfc, i, j)


# Modified from https://github.com/marcellodebernardi/loss-landscapes/blob/8d3461045f317bc0f4ba35e552fb22f3242647ff/loss_landscapes/main.py
def random_plane(model,
                 metric,
                 distance,
                 steps,
                 normalization,
                 dir,
                 sfc,
                 deepcopy_model=False):

    model_start_wrapper = wrap_model(
        copy.deepcopy(model) if deepcopy_model else model)

    start_point = model_start_wrapper.get_module_parameters()
    dir_one = rand_u_like(start_point)
    dir_two = orthogonal_to(dir_one)

    if normalization == 'model':
        dir_one.model_normalize_(start_point)
        dir_two.model_normalize_(start_point)
    elif normalization == 'layer':
        dir_one.layer_normalize_(start_point)
        dir_two.layer_normalize_(start_point)
    elif normalization == 'filter':
        dir_one.filter_normalize_(start_point)
        dir_two.filter_normalize_(start_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument.')

    dir_one.mul_(
        ((start_point.model_norm() / distance) / steps) / dir_one.model_norm())
    dir_two.mul_(
        ((start_point.model_norm() / distance) / steps) / dir_two.model_norm())
    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)

    data_matrix = []
    for i in range(steps):
        data_column = []

        if i % 2 == 0:
            for j in range(steps):
                start_point.add_(dir_two)
                loss = metric(model_start_wrapper)
                data_column.append(loss)
                util.save((loss, start_point), sfcfile(dir, sfc, i, j))
        else:
            for j in range(steps)[::-1]:
                start_point.sub_(dir_two)
                loss = metric(model_start_wrapper)
                data_column.insert(0, loss)
                util.save((loss, start_point), sfcfile(dir, sfc, i, j))

        data_matrix.append(data_column)
        start_point.add_(dir_one)

    return np.array(data_matrix)


def draw_surface(args, sfc, ax):
    model, loader, _, loss_fn = MNIST(**vars(args))
    M.load(model, args.log + "/{}-{}.pkl".format(sfc, args.e))

    x, y = iter(loader).__next__()
    loss_fn = loss_landscapes.metrics.Loss(loss_fn, x, y)
    loss = random_plane(model, loss_fn, args.dist, args.step, "filter",
                        args.log, sfc)

    X = np.array([[j for j in range(args.step)] for i in range(args.step)])
    Y = np.array([[i for _ in range(args.step)] for i in range(args.step)])
    ax.plot_surface(X, Y, loss, alpha=0.8)


def draw_trajectory(args, sfc, traj, ax, color="red"):
    model, _, _, _ = MNIST(**vars(args))
    xline, yline, zline = [], [], []
    for e in range(args.e + 1):
        M.load(model, args.log + "/{}-{}.pkl".format(traj, e))
        mdiff, mi, mj, mloss = np.finfo(np.float32).max, None, None, None
        for i in range(args.step):
            for j in range(args.step):
                (loss, p) = util.load(sfcfile(args.log, sfc, i, j))
                diff = 0
                for x, y in zip(model.parameters(), p.parameters):
                    diff += np.linalg.norm(x.data - y.data)**2
                diff = np.sqrt(diff)
                if diff < mdiff:
                    mdiff, mi, mj, mloss = diff, i, j, loss
        print("{} {}-{} {} {}".format(e, mi, mj, mloss, mdiff))
        xline.append(mi)
        yline.append(mj)
        zline.append(mloss)
    ax.plot3D(xline, yline, zline, label=traj, color=color)


def main(args):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title("Loss Surface")
    draw_surface(args, args.sfc, ax)
    traj = args.traj.split(",")
    for i, x in enumerate(traj):
        draw_trajectory(args, args.sfc, x, ax, random_color(i, len(traj)))
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main(parse_args())
