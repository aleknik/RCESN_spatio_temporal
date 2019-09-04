import os

import numpy as np
from mpi4py import MPI

from esn_parallel import ESNParallel
from mpi_logger import print_with_rank
from utils import load_data, dict_to_string, get_config

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

master_node_rank = 0
shift_count = 10

default_config = {
    'number_of_reservoirs': 11,
    'number_of_features': 88,
    'reservoir_size': 1000,
    'training_size': 100000,
    'prediction_size': 1000,
    'overlap_size': 0,
    'sigma': 0.5,
    'radius': 0.9,
    'beta': 0.0001,
    'degree': 3
}

shifts = list(range(0, default_config['prediction_size'] * shift_count, default_config['prediction_size']))


def main():
    work_root = os.environ['WORK']

    config = get_config(default_config)
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
        all_data = load_data(work_root, 'data/QG_everydt_avgu.csv')
    else:
        all_data = None

    for shift in shifts:
        config['shift'] = shift
        if rank == master_node_rank:
            data = all_data[:, shift: train_length + shift]
        else:
            data = None

        output = ESNParallel(group_count, feature_count, lsp, train_length, predict_length, approx_res_size, radius,
                             sigma, random_state=42, beta=beta, degree=degree).fit(data).predict()

        if rank == master_node_rank:
            shift_folder = dict_to_string({k: v for k, v in config.items() if k != 'shift'})
            directory = os.path.join(work_root, 'results/shift_results', shift_folder)
            if not os.path.exists(directory):
                os.makedirs(directory)
            result_path = os.path.join(directory, 'data=QG-' + dict_to_string(config) + '.txt')
            np.savetxt(result_path, output)
            print_with_rank("Saved to " + result_path)


if __name__ == '__main__':
    main()
