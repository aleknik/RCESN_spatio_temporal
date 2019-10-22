import numpy as np
import scipy.sparse as sparse
from numpy.linalg import eigvals


def generate_reservoir(size, radius, degree, random_state):
    sparsity = degree / float(size)

    A = sparse.rand(size, size, density=sparsity, random_state=random_state).todense()
    vals = eigvals(A)
    e = np.max(np.abs(vals))
    A = (A / e) * radius
    return A


def reservoir_layer(A, Win, input, n, alpha, bias):
    states = np.zeros((n, input.shape[1] + 1))
    for i in range(input.shape[1]):
        if bias:
            input_bias = np.append([1], input[:, i], axis=0)
        else:
            input_bias = input[:, i]
        update = np.tanh(np.dot(A, states[:, i]) + np.dot(Win, input_bias))
        states[:, i + 1] = (1 - alpha) * states[:, i] + alpha * update
    return states


def train(beta, states, data, n, lsp):
    idenmat = beta * sparse.identity(n)
    U = np.dot(states, states.transpose()) + idenmat
    Uinv = np.linalg.inv(U)

    Wout = np.dot(Uinv, np.dot(states, data[lsp:data.shape[0] - lsp, :].transpose())).T
    prediction = np.dot(Wout, states)
    error = np.sum(np.square(prediction - data[lsp:data.shape[0] - lsp, :]))
    return Wout, error

    # Wout = np.dot(np.dot(data[lsp:data.shape[0] - lsp, :], states.transpose()), Uinv)
    # return Wout


class ESN:
    def __init__(self, radius=0.1, degree=3, sigma=0.5, approx_res_size=5000, beta=0.0001, random_state=None, lsp=0,
                 alpha=1, bias=False):
        self._radius = radius
        self._degree = degree
        self._sigma = sigma
        self._approx_res_size = approx_res_size
        self._beta = beta
        self._random_state = random_state
        self._lsp = lsp
        self._alpha = alpha

        self._fn = None
        self._n = None
        self._A = None
        self._Win = None
        self._Wout = None
        self._states = None
        self._data = None
        self._bias = bias

        self.x = None
        self._train_x = None
        self.training_error = 0

    def generate_reservoir(self, data):
        """
        Generate reservoir layer.
        :param data: training data
        :return: generated states
        """
        self._fn = data.shape[0]
        self._n = int(np.floor(self._approx_res_size / self._fn) * self._fn)
        self._A = generate_reservoir(self._n, self._radius, self._degree, self._random_state)

        if self._bias:
            Win_size = self._fn + 1
        else:
            Win_size = self._fn

        q = int(self._n / Win_size)
        self._Win = np.zeros((self._n, Win_size))
        for i in range(Win_size):  # init input layer
            np.random.seed(seed=i)
            self._Win[i * q: (i + 1) * q, i] = self._sigma * (-1 + 2 * np.random.rand(1, q)[0])

        states = reservoir_layer(self._A, self._Win, data, self._n, self._alpha, self._bias)
        self._train_x = states[:, -1].copy()

        return states[:, :-1]

    def transform_states(self, states):
        """
        Add nonlinearity to states.

        :param states: Training states
        :return: transformed states
        """
        for j in list(reversed(range(2, np.shape(states)[0] - 2))):
            if np.mod(j, 2) == 0:
                states[j, :] = states[j - 1, :] * states[j - 2, :]
        return states

    def fit_output(self):
        """
        Fit output layer by ridge regression
        :return: self
        """
        self._Wout, self.training_error = train(self._beta, self._states, self._data, self._n, self._lsp)
        self.x = self._train_x.copy()

        return self

    def fit_reservoir(self, data):
        """
         and states for training data and states for training data.
        :param data: training data
        :return: self
        """
        self._data = data
        self._states = self.generate_reservoir(data)
        self._states = self.transform_states(self._states)
        return self

    def fit(self, data):
        """
        Fit model.
        :param data: training data
        :return: self
        """
        return self.fit_reservoir(data).fit_output()

    def predict(self, predict_length):
        """
        Predict series.
        :param predict_length: number of time steps to predict
        :return: prediction
        """
        output = np.zeros((self._fn, predict_length))
        out = self.predict_next()
        output[:, 0] = out
        for i in range(1, predict_length):
            out = self.predict_next(out)
            output[:, i] = out
        return output

    def predict_next(self, u=None):
        """
        Predict next time step based on internal state and current step.
        :param u: current time step
        :return: predicted time step
        """
        if u is not None:
            if self._bias:
                u = np.append([1], u)
            x1 = np.tanh(np.dot(self._A, self.x) + np.dot(self._Win, u))
            self.x = np.squeeze(np.asarray(x1))
        x_aug = self.x.copy()
        for j in range(2, np.shape(x_aug)[0] - 2):
            if np.mod(j, 2) == 0:
                x_aug[j] = self.x[j - 1] * self.x[j - 2]
        out = np.squeeze(np.asarray(np.dot(self._Wout, x_aug)), axis=0)
        return out
