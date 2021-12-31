import numpy as np
import matplotlib.pyplot as plt


def f1(x, y):
    return 2 / (np.pi * np.pi * 4) * np.cos(np.sqrt(3) * x + y)


def f2(x, y):
    return 2 / (np.pi * np.pi * 4) * np.sin(np.sqrt(3) * x + y)


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z1 = f1(X, Y)
Z2 = f2(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
c1 = ax.contour3D(X, Y, Z1, 50, colors='red', label='f1')
c2 = ax.contour3D(X, Y, Z2, 50, colors='blue', label='f2')
h1, _ = c1.legend_elements()
h2, _ = c2.legend_elements()
ax.legend([h1[0], h2[0]], ['f1', 'f2'])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f1/f2')

plt.show()
