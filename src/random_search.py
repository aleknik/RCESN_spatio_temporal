import os
import random

import numpy as np
from mpi4py import MPI

from esn_parallel import ESNParallel
from mpi_logger import print_with_rank
from utils import load_data, dict_to_string

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

master_node_rank = 0
shift_count = 10


def main():
    param_grid = {
        'group_count': [11],
        'feature_count': [88],
        'lsp': [7],
        'train_length': [100000],
        'predict_length': [1000],
        'approx_res_size': [1000],
        'radius': list(np.linspace(0.0001, 1, endpoint=False, num=1000)),
        'sigma': list(np.linspace(0.0001, 1, num=1000)),
        'random_state': [42],
        'beta': [0.003],
        'degree': [7],
        'alpha': list(np.linspace(0.0001, 1, num=1000)),
    }

    shifts = list(range(0, param_grid['predict_length'][0] * shift_count, param_grid['predict_length'][0]))

    if rank == master_node_rank:
        work_root = os.environ['WORK']
        all_data = load_data(work_root, 'data/QG_everydt_avgu.csv')
    else:
        all_data = None
        work_root = None

    max_evals = 500000
    for i in range(max_evals):

        if rank == master_node_rank:
            params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            print_with_rank(str(params))
        else:
            params = None

        params = comm.bcast(params, master_node_rank)

        for shift in shifts:
            if rank == master_node_rank:
                data = all_data[:, shift: params['train_length'] + shift]
            else:
                data = None

            output = ESNParallel(**params).fit(data).predict()

            if rank == master_node_rank:
                params['shift'] = shift
                shift_folder = dict_to_string({k: v for k, v in params.items() if k != 'shift'})
                directory = os.path.join(work_root, 'results/random_shift_results', shift_folder)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                result_path = os.path.join(directory, 'data=QG-' + dict_to_string(params) + '.txt')
                np.savetxt(result_path, output)
                print_with_rank("Saved to " + result_path)
                del params['shift']


if __name__ == '__main__':
    main()
