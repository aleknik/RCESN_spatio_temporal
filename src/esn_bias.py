import numpy as np
import scipy.sparse as sparse
from numpy.linalg import eigvals

from mpi_logger import print_with_rank


def generate_reservoir(size, radius, degree, random_state):
    sparsity = degree / float(size)

    A = sparse.rand(size, size, density=sparsity, random_state=random_state).todense()
    vals = eigvals(A)
    e = np.max(np.abs(vals))
    A = (A / e) * radius
    return A


def reservoir_layer(A, Win, input, n):
    states = np.zeros((n, input.shape[1]))
    for i in range(input.shape[1] - 1):
        states[:, i + 1] = np.tanh(np.dot(A, states[:, i]) + np.dot(Win, np.append([1], input[:, i], axis=0)))
    return states


def train(beta, states, data, n, lsp):
    idenmat = beta * sparse.identity(n)
    for j in list(reversed(range(2, np.shape(states)[0] - 2))):
        if np.mod(j, 2) == 0:
            states[j, :] = states[j - 1, :] * states[j - 2, :]
    U = np.dot(states, states.transpose()) + idenmat
    Uinv = np.linalg.inv(U)

    Wout = np.dot(Uinv, np.dot(states, data[lsp:data.shape[0] - lsp, :].transpose()))
    return Wout.transpose()

    # Wout = np.dot(np.dot(data[lsp:data.shape[0] - lsp, :], states.transpose()), Uinv)
    # return Wout


class ESN:
    def __init__(self, radius=0.1, degree=3, sigma=0.5, approx_res_size=5000, beta=0.0001, random_state=None, lsp=0):
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
        print_with_rank('Reservoir generated')

        Win_size = self._fn + 1

        q = int(self._n / Win_size)
        self._Win = np.zeros((self._n, Win_size))
        for i in range(Win_size):  # init input layer
            np.random.seed(seed=i)
            self._Win[i * q: (i + 1) * q, i] = self._sigma * (-1 + 2 * np.random.rand(1, q)[0])

        states = reservoir_layer(self._A, self._Win, data, self._n)
        print_with_rank('States generated')
        self.x = states[:, -1].copy()
        self._Wout = train(self._beta, states, data, self._n, self._lsp)
        print_with_rank('Training outputs finished')
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
            x1 = np.tanh(np.dot(self._A, self.x) + np.dot(self._Win, np.append([1], u)))
            self.x = np.squeeze(np.asarray(x1))
        x_aug = self.x.copy()
        for j in range(2, np.shape(x_aug)[0] - 2):
            if np.mod(j, 2) == 0:
                x_aug[j] = self.x[j - 1] * self.x[j - 2]
        out = np.squeeze(np.asarray(np.dot(self._Wout, x_aug)), axis=0)
        return out
