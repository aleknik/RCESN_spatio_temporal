import os

import numpy as np
import pandas as pd
from mpi4py import MPI

from esn_parallel import ESNParallel

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

master_node_rank = 0

config = {
    'number_of_reservoirs': 11,
    'number_of_features': 88,
    'reservoir_size': 1000,
    'training_size': 50000,
    'prediction_size': 1000,
    'overlap_size': 6,
    'sigma': 0.2,
    'radius': 0.9
}


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
    pd_data = pd.read_csv(work_root + '/data/QG_everydt_avgu.csv', header=None)
    pd_data = standardize_data(pd_data)

    return np.array(pd_data)[:, :train_length]


def main():
    group_grid = [11, 22]
    res_grid = [5000]
    lsp_grid = [6, 9, 12]
    sigma_grid = [0.5]
    radius_grid = [0.9, 0.5, 0.2]

    work_root = os.environ['WORK']

    # Read hyper parameters
    group_count = config['number_of_reservoirs']
    feature_count = config['number_of_features']
    lsp = config['overlap_size']
    predict_length = config['prediction_size']
    train_length = config['training_size']
    approx_res_size = config['reservoir_size']
    sigma = config['sigma']
    radius = config['radius']

    if rank == master_node_rank:
        data = load_data(train_length, work_root)
    else:
        data = None

    for group in group_grid:
        for res in res_grid:
            for lsp in lsp_grid:
                for sigma in sigma_grid:
                    for radius in radius_grid:
                        config['number_of_reservoirs'] = group
                        config['reservoir_size'] = res
                        config['overlap_size'] = lsp
                        config['sigma'] = sigma
                        config['radius'] = radius

                        output = ESNParallel(group, feature_count, lsp, train_length, predict_length, res, radius,
                                             sigma, random_state=42).fit(data).predict()

                        if rank == master_node_rank:
                            result_path = work_root + '/grid/GRID-QG' + dict_to_string(config) + '.txt'
                            np.savetxt(result_path, output)


if __name__ == '__main__':
    main()
