import os

import numpy as np
import pandas as pd
from mpi4py import MPI

import arg_parser
from esn_parallel import ESNParallel
from mpi_logger import print_with_rank

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

master_node_rank = 0

config = {
    'number_of_reservoirs': 1,
    'number_of_features': 88,
    'reservoir_size': 500,
    'training_size': 10000,
    'prediction_size': 100,
    'overlap_size': 0,
    'sigma': 0.5,
    'radius': 0.9,
    'beta': 0.0001,
    'degree': 3
}


def dict_to_string(dict):
    string = ''
    for key, val in sorted(dict.items()):
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


def get_config():
    names = ['g', 'l', 'r', 's', 'rad', 'b', 'd']

    args = arg_parser.parse(names)

    for arg, value in args.items():
        if arg == 'g' and value is not None:
            config['number_of_reservoirs'] = int(value)
        elif arg == 'l' and value is not None:
            config['overlap_size'] = int(value)
        elif arg == 'r' and value is not None:
            config['reservoir_size'] = int(value)
        elif arg == 's' and value is not None:
            config['sigma'] = float(value)
        elif arg == 'rad' and value is not None:
            config['radius'] = float(value)
        elif arg == 'b' and value is not None:
            config['beta'] = float(value)
        elif arg == 'd' and value is not None:
            config['degree'] = int(value)
    return config


def main():
    work_root = os.environ['WORK']

    config = get_config()
    print_with_rank(str(config))

    # Read hyper parameters
    group_count = config['number_of_reservoirs']
    feature_count = config['number_of_features']
    lsp = config['overlap_size']
    predict_length = config['prediction_size']
    train_length = config['training_size']
    approx_res_size = config['reservoir_size']
    sigma = config['sigma']
    radius = config['radius']
    beta = config['beta']
    degree = config['degree']

    # Check preconditions for running
    if feature_count % group_count != 0:
        comm.Abort()
        return
    if group_count % size != 0:
        comm.Abort()
        return

    if rank == master_node_rank:
        data = load_data(train_length, work_root)
    else:
        data = None

    output = ESNParallel(group_count, feature_count, lsp, train_length, predict_length, approx_res_size, radius,
                         sigma, random_state=42, beta=beta, degree=degree).fit(data).predict()

    if rank == master_node_rank:
        result_path = work_root + '/results/Ridge_SCL_QG' + dict_to_string(config) + '.txt'
        np.savetxt(result_path, output)


if __name__ == '__main__':
    main()
