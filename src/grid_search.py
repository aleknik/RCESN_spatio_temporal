import itertools
import os

import numpy as np
import pandas as pd
from mpi4py import MPI

from esn_parallel import ESNParallel
from mpi_logger import print_with_rank

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

master_node_rank = 0
shift = 0


def dict_to_string(dict):
    string = ''
    for key, val in dict.items():
        string += '_' + str(key) + '-' + str(val)

    return string


def standardize_data(data):
    data = data.T
    data = (data - data.mean()) / data.std()
    return data.T


def load_data(train_length, work_root):
    # pd_data = pd.read_csv(work_root + '/data/3tier_lorenz_v3.csv', header=None).T
    # pd_data = pd.read_csv(work_root + '/data/ks_64.csv', header=None)
    print(work_root, train_length)
    pd_data = pd.read_csv(work_root + '/data/QG_everydt_avgu.csv', header=None)
    pd_data = standardize_data(pd_data)

    return np.array(pd_data)[:, shift:train_length + shift]


def main():
    param_grid = {
        'group_count': [11],
        'feature_count': [88],
        'lsp': list(range(3, 30)),
        'train_length': [100000],
        'predict_length': [1000],
        'approx_res_size': [1000],
        'radius': [0.95],
        'sigma': [0.08],
        'random_state': [42],
        'beta': [0.003],
        'degree': [7]
    }

    if rank == master_node_rank:
        work_root = os.environ['WORK']
        data = load_data(param_grid['train_length'][0], work_root)
    else:
        data = None
        work_root = None

    keys, values = zip(*param_grid.items())

    # Iterate through every possible combination of hyperparameters
    for v in itertools.product(*values):
        # Create a hyperparameter dictionary
        params = dict(zip(keys, v))
        print_with_rank(str(params))

        output = ESNParallel(**params).fit(data).predict()

        if rank == master_node_rank:
            result_path = work_root + '/grid1000-11/RANDOM-QG' + dict_to_string(params) + '.txt'
            np.savetxt(result_path, output)


if __name__ == '__main__':
    main()
