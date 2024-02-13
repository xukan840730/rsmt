import numpy as np


def test1():
    lp = np.zeros((2, 4, 3))
    for i in range(2):
        for j in range(4):
            for k in range(3):
                lp[i, j, k] = 100 * i + 10 * j + k

    a = lp[:, 1:]
    b = lp[:, :-1]
    relative_gv = a - b
    print(relative_gv)


if __name__ == '__main__':
    test1()
