import cvxpy as cvx
import numpy as np

np.random.seed(214)

def p1():
    t = cvx.Variable()
    h = cvx.Variable()
    v = 1

    objective = cvx.Minimize(2*np.pi*t*t+2*np.pi*t*h)
    constraints = [np.pi*t*t*h==v]

    prob = cvx.Problem(objective, constraints)
    result = prob.solve(solver=cvx.CVXOPT)
    print(t.value, h.value)


def main():
    p1()

if __name__ == "__main__":
    main()
