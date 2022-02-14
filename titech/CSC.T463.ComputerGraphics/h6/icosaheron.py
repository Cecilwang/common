import math


def dist(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)


def polygon(r, x, y, z, n, theta=0.):
    points = []
    for i in range(n):
        points += [(x + r * math.cos(theta), y + r * math.sin(theta), z)]
        theta += math.pi * 2 / n
    return points


def icosahedron(r, x, y, z):
    p1 = polygon(r * math.sqrt(5) * 2 / 5, x, y, z - r * math.sqrt(5), 5)
    print(dist([x, y, z - r], p1[0]))
    print(dist([x, y, z - r], p1[1]))
    print(dist(p1[0], p1[1]))
    print(dist(p1[1], p1[2]))
    p2 = polygon(r * math.sqrt(5) * 2 / 5, x, y, z + r * math.sqrt(5), 5,
                 36 / 360 * 2 * math.pi)


icosahedron(1, 0, 0, 0)
