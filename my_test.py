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


def test2():
    import pickle
    f = open('output/fk_res_01/Aeroplane_BR_fk_res.dat', 'rb')
    dict = pickle.load(f)
    f.close()


if __name__ == '__main__':
    # test1()
    test2()
