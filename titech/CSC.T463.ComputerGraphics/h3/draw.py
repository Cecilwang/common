import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.stats as st


def parse_args():
    parser = argparse.ArgumentParser(description="prnning")
    parser.add_argument("--f", type=int, default=None)
    args = parser.parse_args()
    args.inv = 512 // args.f
    return args


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def main():
    args = parse_args()

    #img = np.zeros((512, 512))
    #black = 0
    #white = 255
    #u = black
    #v = black
    #for i in range(0, 512, args.inv):
    #    v = u
    #    for j in range(0, 512, args.inv):
    #        img[i:i + args.inv, j:j + args.inv] = v
    #        v = v ^ black ^ white
    #    u = u ^ black ^ white
    #img = img.astype(np.uint8)
    #print(img)
    #Image.fromarray(img).save(f"/tmp/{args.f}.png")

    n = 3
    kernel = gkern(n, 1)
    fkernel = np.fft.fft2(kernel)
    fkernel = np.fft.fftshift(fkernel)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(0, n, 1)
    X, Y = np.meshgrid(X, X)
    surf = ax.plot_surface(X, Y, kernel, label='signal')
    surf = ax.plot_surface(X + n, Y + n, np.abs(fkernel), label='frequency')
    plt.show()
    print(kernel)
    print(np.abs(fkernel))
    img = (kernel * 255).astype(np.uint8)
    print(img)
    Image.fromarray(img).save(
        f"/Users/cecilwang/Workspace/TestCode/FreqExperiments/images/g{n}.png")


if __name__ == "__main__":
    main()
