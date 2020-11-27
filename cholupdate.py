#!/usr/bin/python3

import numpy as np


def cholupdate(L, x, n):
    """
    choleskey decomposition, rank one update, in-place process

    :param L: L matrix, size(n, n)
    :param x: x, input vector, (n, 1)
    :param n: element size
    """
    for k in range(0, n):
        r = np.sqrt(L[k, k] * L[k, k].conj() + x[k, 0] * x[k, 0].conj())
        c = r / L[k, k]
        s = x[k, 0] / L[k, k]
        L[k, k] = r

        L[k+1:n, k] = (L[k+1:n, k] + s.conj() * x[k+1:n, 0]) / c
        x[k+1:n, 0] = c * x[k+1:n, 0] - s * L[k+1:n, k]

    return L


def ldlh_update(L, D, x, n):
    """
    A = (1-alpha)*A + alpha * x * x^H
    """
    # a = 0.1
    a = 1
    for k in range(0, n):
        # d = (1 - a) * D[k, k] + a * x[k, 0] * x[k, 0].conj()
        d = D[k, k] + a * x[k, 0] * x[k, 0].conj()
        b = x[k, 0] * a / d
        # a = (1 - a) * D[k, k] * a / d
        a = D[k, k] * a / d
        D[k, k] = d

        for r in range(k+1, n):
            x[r, 0] = x[r, 0] - x[k, 0] * L[r, k]
            L[r, k] = L[r, k] + b.conj() * x[r, 0]

    return L, D


def forward_substitution(L, b, n):
    """
    Lx = b, solve x
    """
    x = np.zeros((n, n), dtype=complex)
    for j in range(0, n):
        for i in range(j, n):
            x[i, j] = b[i, j] / L[i, i]
            # b[i+1:n, j] = b[i+1:n, j] - L[i+1:n, i] * x[i, j]
            for k in range(i+1, n):
                b[k, j] = b[k, j] - L[k, i] * x[i, j]

    return x


if __name__ == '__main__':
    num_channel = 3
    D = np.eye(num_channel, dtype=complex) / np.sqrt(num_channel)
    L = np.eye(num_channel, dtype=complex)
    x = np.zeros((num_channel, 1), dtype=complex)

    for i in range(0, num_channel):
        D[i, i] = np.random.rand()
        for j in range(0, i):
            L[i, j] = np.random.rand() + 1.j * np.random.rand()

    num_iteration = 1
    for k in range(0, num_iteration):
        for i in range(0, num_channel):
            x[i, 0] = np.random.rand() + 1.j * np.random.rand()
        x[:, 0] = x[:, 0] / np.sqrt(np.matmul(x.conj().T, x))

        xx = x.copy()
        LL = L.copy()

        """
        # print("before")
        # print("x:\n {}".format(x))
        xxH = np.matmul(x,x.conj().T)
        LLH = np.matmul(L, L.conj().T)
        # print("xx^H:\n {}".format(xxH))
        # print("LL^H:\n {}".format(LLH))
        ground = LLH + xxH
        print("ground:\n {}".format(ground))

        L = cholupdate(L, x, num_channel)
        # print("L:\n {}".format(L))
        A = np.matmul(L, L.conj().T)
        print("A:\n {}".format(A))
        """

        xxH = np.matmul(x, x.conj().T)
        LDLH = np.matmul(L, np.matmul(D, L.conj().T))

        # ground = 0.9*LDLH + 0.1*xxH
        ground = LDLH + xxH
        print("ground:\n {}".format(ground))

        L, D = ldlh_update(L, D, x, num_channel)
        A = np.matmul(L, np.matmul(D, L.conj().T))
        print("A:\n {}".format(A))

        D_inv = D.copy()
        for i in range(0, num_channel):
            D_inv[i, i] = 1 / D[i, i]

        L_inv = forward_substitution(L, np.eye(num_channel, dtype=complex), num_channel)
        L_inv = L_inv.conj().T
        A_inv = np.matmul(L_inv, np.matmul(D_inv, L_inv.conj().T))

        I = np.matmul(ground, A_inv)

        print("AA^-1:\n {}".format(I))
        print(np.real(np.trace(I)))
