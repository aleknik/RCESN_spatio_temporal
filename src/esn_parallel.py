import numpy as np
from mpi4py import MPI

from esn import ESN

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

master_node_rank = 0


def split_modulo(start, stop, array_len):
    if stop <= start:
        stop += array_len
    return np.arange(start, stop) % array_len


class ESNParallel:

    def __init__(self, group_count, feature_count, lsp, train_length, predict_length, approx_res_size, radius, sigma,
                 random_state, beta=0.0001, degree=3):
        self._lsp = lsp
        self._feature_count = feature_count
        self._group_count = group_count
        self._ftr_per_grp = int(self._feature_count / self._group_count)
        self._res_per_task = int(self._group_count / size)
        self._train_length = train_length
        self._predict_length = predict_length
        self._approx_res_size = approx_res_size
        self._radius = radius
        self._sigma = sigma
        self._output = None
        self._fitted_models = None
        self._n = self._ftr_per_grp + 2 * self._lsp
        self._random_state = random_state
        self._beta = beta
        self._degree = degree

    def fit(self, data):
        if rank == master_node_rank:
            # data = load_data(train_length, work_root)
            splits = np.concatenate(list(
                map(lambda i: data[
                              split_modulo(i * self._ftr_per_grp - self._lsp, (i + 1) * self._ftr_per_grp + self._lsp,
                                           self._feature_count), :],
                    range(self._group_count))))
            self._output = np.zeros((self._feature_count, self._predict_length))
            # print_with_rank('Data loaded')
        else:
            splits = None

        # if rank == master_node_rank:
        #     run_time = MPI.Wtime()

        # Scatter data to each task
        data = np.empty([(self._ftr_per_grp + 2 * self._lsp) * self._res_per_task, self._train_length])
        comm.Scatter(splits, data, root=master_node_rank)

        # print_with_rank('Training started')

        # Split data based on number of reservoirs per task
        data = [data[i * self._n:(i + 1) * self._n, :] for i in range((len(data) + self._n - 1) // self._n)]

        # Fit each model on part of data
        self._fitted_models = list(
            map(lambda x: ESN(lsp=self._lsp, approx_res_size=self._approx_res_size, radius=self._radius,
                              sigma=self._sigma, random_state=self._random_state * (rank + 1), beta=self._beta,
                              degree=self._degree).fit(x),
                data))

        return self

    def predict(self):
        input_parts = [None] * self._res_per_task
        for j in range(self._predict_length):
            # Predict next time step for each model in current task
            output_parts = np.concatenate(
                list(map(lambda model, input_data: model.predict_next(input_data), self._fitted_models, input_parts)))

            # Debug print
            # if j % 100 == 0:
            #     print_with_rank('predicted ' + str(j))

            # Gather all predictions on master task
            if rank == master_node_rank:
                prediction_parts = np.empty([size, self._ftr_per_grp * self._res_per_task])
            else:
                prediction_parts = None
            comm.Gather(output_parts, prediction_parts, root=master_node_rank)

            # Save current prediction on master and split data for next prediction
            if rank == master_node_rank:
                self._output[:, j] = np.concatenate(prediction_parts)
                input_parts_all = np.concatenate(
                    list(map(lambda i: self._output[
                        split_modulo(i * self._ftr_per_grp - self._lsp, (i + 1) * self._ftr_per_grp + self._lsp,
                                     self._feature_count), j],
                             range(self._group_count))))
            else:
                input_parts_all = None

            # Scatter data for next prediction and split is among reservoirs for current task
            input_parts = np.empty((self._ftr_per_grp + 2 * self._lsp) * self._res_per_task)
            comm.Scatter(input_parts_all, input_parts, root=master_node_rank)
            input_parts = [input_parts[i * self._n:(i + 1) * self._n] for i in
                           range((len(input_parts) + self._n - 1) // self._n)]

        return self._output