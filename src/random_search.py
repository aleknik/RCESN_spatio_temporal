import os
import random

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
        'lsp': [14],
        'train_length': [100000],
        'predict_length': [1000],
        'approx_res_size': [1000],
        'radius': list(np.linspace(0.9, 1, endpoint=False, num=10000)),
        'sigma': list(np.linspace(0.01, 0.1, num=10000)),
        'random_state': [42],
        'beta': list(np.linspace(0.001, 1, num=10000)),
        'degree': [7]
    }

    if rank == master_node_rank:
        work_root = os.environ['WORK']
        data = load_data(param_grid['train_length'][0], work_root)
    else:
        data = None
        work_root = None

    MAX_EVALS = 500000
    for i in range(MAX_EVALS):

        if rank == master_node_rank:
            params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            print_with_rank(str(params))
        else:
            params = None

        params = comm.bcast(params, master_node_rank)

        output = ESNParallel(**params).fit(data).predict()

        if rank == master_node_rank:
            result_path = work_root + '/grid1000-fine/RANDOM-QG' + dict_to_string(params) + '.txt'
            np.savetxt(result_path, output)


if __name__ == '__main__':
    main()