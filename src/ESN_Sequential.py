#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from numpy.linalg import eigvals

work_root = os.environ['WORK']


# ESN code
def generate_reservoir(size, radius, degree, random_state):
    sparsity = degree / float(size)

    A = sparse.rand(size, size, density=sparsity, random_state=random_state).todense()
    vals = eigvals(A)
    e = np.max(np.abs(vals))

    # A = sparse.rand(size, size, density=sparsity, random_state=random_state)
    # vals = eigs(A, k=1, which='LM', return_eigenvectors=False)
    # e = np.abs(vals[0])
    # A = A.todense()

    A = (A / e) * radius
    return A


def reservoir_layer(A, Win, input, n):
    states = np.zeros((n, input.shape[1]))
    for i in range(input.shape[1] - 1):
        states[:, i + 1] = np.tanh(np.dot(A, states[:, i]) + np.dot(Win, input[:, i]))
    return states


def train(beta, states, data, n, lsp):
    idenmat = beta * sparse.identity(n)
    states2 = states.copy()
    for j in range(2, np.shape(states2)[0] - 2):
        if np.mod(j, 2) == 0:
            states2[j, :] = states[j - 1, :] * states[j - 2, :]
    U = np.dot(states2, states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(states2, data[lsp:data.shape[0] - lsp, :].transpose()))
    return Wout.transpose()


class ESN:
    def __init__(self, radius=0.1, degree=3, sigma=0.5, approx_res_size=5000, beta=0.0001, random_state=360, lsp=0):
        self._radius = radius
        self._degree = degree
        self._sigma = sigma
        self._approx_res_size = approx_res_size
        self._beta = beta
        self._random_state = random_state
        self._lsp = lsp

        self._fn = None
        self._n = None
        self._A = None
        self._Win = None
        self._Wout = None

        self.x = None

    def fit(self, data):
        self._fn = data.shape[0]
        self._n = int(np.floor(self._approx_res_size / self._fn) * self._fn)
        self._A = generate_reservoir(self._n, self._radius, self._degree, self._random_state)

        q = int(self._n / self._fn)
        self._Win = np.zeros((self._n, self._fn))
        for i in range(self._fn):  # init input layer
            np.random.seed(seed=i)
            self._Win[i * q: (i + 1) * q, i] = self._sigma * (-1 + 2 * np.random.rand(1, q)[0])

        states = reservoir_layer(self._A, self._Win, data, self._n)
        self._Wout = train(self._beta, states, data, self._n, self._lsp)
        self.x = states[:, -1]
        return self

    def predict(self, predict_length):

        output = np.zeros((self._fn, predict_length))
        out = self.predict_next()
        output[:, 0] = out
        for i in range(1, predict_length):
            out = self.predict_next(out)
            output[:, i] = out
        return output

    def predict_next(self, u=None):
        if u is not None:
            x1 = np.tanh(np.dot(self._A, self.x) + np.dot(self._Win, u))
            self.x = np.squeeze(np.asarray(x1))
        x_aug = self.x.copy()
        for j in range(2, np.shape(x_aug)[0] - 2):
            if np.mod(j, 2) == 0:
                x_aug[j] = self.x[j - 1] * self.x[j - 2]
        out = np.squeeze(np.asarray(np.dot(self._Wout, x_aug)), axis=0)
        return out


def split_modulo(start, stop, array_len):
    if stop <= start:
        stop += array_len
    return np.arange(start, stop) % array_len


def load_data(train_length):
    pd_data = pd.read_csv(work_root + '/data/3tier_lorenz_v3.csv', header=None).T
    print(pd_data.shape)
    return np.array(pd_data)[:, :train_length]


def main():
    Q = 8
    g = 8
    q = int(Q / g)
    lsp = 3
    predict_length = 10000
    train_length = 500000
    approx_res_size = 5000

    data = load_data(train_length)

    splits = list(map(lambda i: data[split_modulo(i * q - lsp, (i + 1) * q + lsp, Q), :], range(g)))

    fitted_models = list(map(lambda x: ESN(lsp=lsp, approx_res_size=approx_res_size).fit(x), splits))

    output_parts = list(map(lambda model: model.predict_next(), fitted_models))

    output = np.zeros((Q, predict_length))
    output[:, 0] = np.concatenate(output_parts)

    input_parts = np.empty(g, dtype=object)
    for j in range(predict_length):
        output_parts = list(map(lambda model, input_part: model.predict_next(input_part), fitted_models, input_parts))
        output[:, j] = np.concatenate(output_parts)
        input_parts = list(map(lambda i: output[split_modulo(i * q - lsp, (i + 1) * q + lsp, Q), j], range(g)))

    print(output.shape)
    np.savetxt(work_root + '/data/Sequential_Expansion_2step_back_' + str(g) + '.txt', output)


if __name__ == '__main__':
    main()
