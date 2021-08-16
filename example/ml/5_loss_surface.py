import argparse
import copy
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import torch

from common.py.ml.util import models as M
from common.py.ml.util.problems import MNIST

Color = cycle("bgrcmk")


def parse_args():
    parser = argparse.ArgumentParser(description="loss_surface")
    parser.add_argument("--model",
                        type=str,
                        default="cnn",
                        choices=["mlp", "cnn"])
    parser.add_argument("sfc", type=str)
    parser.add_argument("traj", type=str)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data", type=str, default="./.data")
    parser.add_argument("--log", type=str, default="./.log")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--decomposition", type=str, default="svd")
    parser.add_argument("--matrix", type=str, default="diff")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.3)
    return parser.parse_args()


def random_color():
    return next(Color)


class Vector(object):

    def __init__(self, arg):
        if isinstance(arg, list):
            params = arg
            self.data = torch.cat([p.view(p.numel()) for p in params])
            self.data = self.data.numpy().astype(np.float32)
        elif isinstance(arg, np.ndarray):
            self.data = arg.astype(np.float32)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return Vector(self.data + other.data)

    def __sub__(self, other):
        return Vector(self.data - other.data)

    def __mul__(self, scalar):
        return Vector(self.data * scalar)

    def dot(self, other):
        return np.dot(self.data, other.data)

    def norm(self):
        return np.linalg.norm(self.data)

    def angle(self, other):
        return self.dot(other) / (self.norm() * other.norm())

    def project(self, space):
        x = self.dot(space.d1) / space.d1.norm()
        y = self.dot(space.d2) / space.d2.norm()
        return Point(x, y, 0)

    def assign_to(self, model):
        i = 0
        for p in model.parameters():
            p.data = torch.tensor(self[i:i + p.numel()]).view(p.size())
            i += p.numel()
        assert i == len(self)


class Space(object):

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def project(self, point):
        return self.d1 * point.x + self.d2 * point.y


class Point(object):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.z)


class Contour(object):

    def __init__(self):
        self.u = 0
        self.b = 0
        self.l = 0
        self.r = 0

    def update(self, point):
        self.u = min(self.u, point.x)
        self.b = max(self.b, point.x)
        self.l = min(self.l, point.y)
        self.r = max(self.r, point.y)

    def round(self):
        self.u = np.int32(self.u)
        self.b = np.int32(self.b)
        self.l = np.int32(self.l)
        self.r = np.int32(self.r)

    def __str__(self):
        return "({}, {}), ({}, {})".format(self.u, self.l, self.b, self.r)

    def __imul__(self, scalar):
        self.u -= (self.b - self.u) * scalar
        self.b += (self.b - self.u) * scalar
        self.l -= (self.r - self.l) * scalar
        self.r += (self.r - self.l) * scalar
        return self


class Trajectory(object):

    def __init__(self, name):
        self.name = name
        self.points = []

    def append(self, point):
        self.points.append(point)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return self.points[i]


def model_dist(x, y):
    x = Vector([p.data for p in x.parameters()])
    y = Vector([p.data for p in y.parameters()])
    return x - y


def get_space(final_model, traj_files, matrix_type, decomposition_method):
    matrix = []
    curr_model = copy.deepcopy(final_model)
    for _, model_files in traj_files.items():
        for f in model_files:
            M.load(curr_model, f)
            if matrix_type == "diff":
                matrix.append(model_dist(curr_model, final_model).data)
            elif matrix_type == "original":
                column = Vector([p.data for p in curr_model.parameters()]).data
                matrix.append(column)
            else:
                raise NotImplementedError

    if decomposition_method == "pca":
        method = decomposition.PCA(n_components=2)
    elif decomposition_method == "svd":
        method = decomposition.TruncatedSVD(n_components=2)
    else:
        #method = decomposition.FactorAnalysis(n_components=2)
        #method = decomposition.FastICA(n_components=2)
        raise NotImplementedError

    method.fit(np.array(matrix))
    d1 = Vector(np.array(method.components_[0]))
    d2 = Vector(np.array(method.components_[1]))
    print("Angle between directions: {}".format(d1.angle(d2)))
    return Space(d1, d2)


def project(final_model, traj_files, space, loss_fn):
    trajectories = {}
    curr_model = copy.deepcopy(final_model)
    for name, model_files in traj_files.items():
        trajectories[name] = Trajectory(name)
        for f in model_files:
            M.load(curr_model, f)
            p = model_dist(curr_model, final_model).project(space)
            p.z = loss_fn(curr_model)
            d = (space.project(p) - model_dist(curr_model, final_model)).norm()
            print("{} at {} {}".format(f, p, d))
            trajectories[name].append(p)
    return trajectories


def get_surface_scope(trajectories, margin, step):
    contour = Contour()
    for _, traj in trajectories.items():
        for i in range(len(traj)):
            contour.update(traj[i])
    contour *= margin
    contour.round()
    print(contour)

    x_stride = (contour.b - contour.u) // step
    y_stride = (contour.r - contour.l) // step
    X = [i for i in range(contour.u, contour.b + 1, x_stride)]
    Y = [i for i in range(contour.l, contour.r + 1, y_stride)]
    return X, Y


def get_surface(final_model, loss_fn, space, scope):
    curr_model = copy.deepcopy(final_model)
    center = Vector([p.data for p in curr_model.parameters()])
    X, Y = scope[0], scope[1]
    loss = np.ndarray([len(X), len(Y)])
    for i in range(len(X)):
        for j in range(len(Y)):
            (space.project(Point(X[i], Y[j], 0)) + center).assign_to(curr_model)
            loss[i, j] = loss_fn(curr_model)

    Y, X = np.meshgrid(Y, X)
    return X, Y, loss


def draw(surface, trajectories):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.set_title("Loss Surface")
    ax.plot_surface(*surface, alpha=0.8)

    for name, traj in trajectories.items():
        x = [traj[i].x for i in range(len(traj))]
        y = [traj[i].y for i in range(len(traj))]
        z = [traj[i].z for i in range(len(traj))]
        ax.plot3D(x, y, z, label=name, color=random_color())
    ax.legend()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("loss")

    plt.show()


def wrap_loss_fn(loss_fn, loader, device):

    def _loss(model):
        loss = 0.0
        for i, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss += loss_fn(output, target).item()
            break
        return loss / (i + 1)

    return _loss


def get_traj_files(args):
    files = {}
    for x in args.traj.split(","):
        traj = []
        for i in range(args.e + 1):
            traj.append("{}/{}-{}-{}.pkl".format(args.log, args.model, x, i))
        files[x] = traj
    return files


def main(args):
    final_model, loader, _, _loss_fn = MNIST(**vars(args))
    final_model.eval()
    M.load(final_model, "{}/{}-{}-{}.pkl".format(args.log, args.model, args.sfc,
                                                 args.e))
    loss_fn = wrap_loss_fn(_loss_fn, loader, args.device)

    traj_files = get_traj_files(args)
    space = get_space(final_model, traj_files, args.matrix, args.decomposition)
    trajectories = project(final_model, traj_files, space, loss_fn)
    scope = get_surface_scope(trajectories, args.margin, args.step)
    surface = get_surface(final_model, loss_fn, space, scope)

    draw(surface, trajectories)


if __name__ == "__main__":
    main(parse_args())
