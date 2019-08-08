#!/usr/bin/env python
# coding: utf-8

# mpiexec -n 2 python esn_parallel.py

import os

import numpy as np
import pandas as pd
from mpi4py import MPI

from src import arg_parser
from src.esn import ESN
from mpi_logger import print_with_rank

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

master_node_rank = 0

config = {
    'number_of_reservoirs': 11,
    'number_of_features': 88,
    'reservoir_size': 500,
    'training_size': 5000,
    'prediction_size': 100,
    'overlap_size': 6,
    'sigma': 1,
    'radius': 0.9
}


def dict_to_string(dict):
    string = ''
    for key, val in dict.items():
        string += '_' + str(key) + '_' + str(val)

    return string


def split_modulo(start, stop, array_len):
    if stop <= start:
        stop += array_len
    return np.arange(start, stop) % array_len


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
    names = ['g', 'l', 'r']

    args = arg_parser.parse(names)

    for arg, value in args.items():
        if arg == 'g' and value is not None:
            config['number_of_reservoirs'] = int(value)
        elif arg == 'l' and value is not None:
            config['overlap_size'] = int(value)
        elif arg == 'r' and value is not None:
            config['reservoir_size'] = int(value)
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

    # Check preconditions for running
    if feature_count % group_count != 0:
        comm.Abort()
        return
    if group_count % size != 0:
        comm.Abort()
        return

    ftr_per_grp = int(feature_count / group_count)
    res_per_task = int(group_count / size)

    # run_time = 0.0

    # Load data on master task and split it based on number of reservoirs and overlap size
    if rank == master_node_rank:
        data = load_data(train_length, work_root)
        splits = np.concatenate(list(
            map(lambda i: data[split_modulo(i * ftr_per_grp - lsp, (i + 1) * ftr_per_grp + lsp, feature_count), :],
                range(group_count))))
        output = np.zeros((feature_count, predict_length))
        print_with_rank('Data loaded')
    else:
        splits = None
        output = None

    # if rank == master_node_rank:
    #     run_time = MPI.Wtime()

    # Scatter data to each task
    data = np.empty([(ftr_per_grp + 2 * lsp) * res_per_task, train_length])
    comm.Scatter(splits, data, root=master_node_rank)

    # print_with_rank('Training started')

    # Split data based on number of reservoirs per task
    n = ftr_per_grp + 2 * lsp
    data = [data[i * n:(i + 1) * n, :] for i in range((len(data) + n - 1) // n)]

    # Fit each model on part of data
    fitted_models = list(
        map(lambda x: ESN(lsp=lsp, approx_res_size=approx_res_size, radius=radius, sigma=sigma).fit(x), data))
    # print_with_rank('Training finished')

    # comm.barrier()
    # if rank == master_node_rank:
    #     print_with_rank('Training time: ' + str(MPI.Wtime() - run_time) + ' Size: ' + str(size))
    #
    # return

    # Start predicting
    # print_with_rank('Prediction started')
    input_parts = [None] * res_per_task
    for j in range(predict_length):
        # Predict next time step for each model in current task
        output_parts = np.concatenate(
            list(map(lambda model, input_data: model.predict_next(input_data), fitted_models, input_parts)))

        # Debug print
        # if j % 100 == 0:
        #     print_with_rank('predicted ' + str(j))

        # Gather all predictions on master task
        if rank == master_node_rank:
            prediction_parts = np.empty([size, ftr_per_grp * res_per_task])
        else:
            prediction_parts = None
        comm.Gather(output_parts, prediction_parts, root=master_node_rank)

        # Save current prediction on master and split data for next prediction
        if rank == master_node_rank:
            output[:, j] = np.concatenate(prediction_parts)
            input_parts_all = np.concatenate(
                list(map(lambda i: output[
                    split_modulo(i * ftr_per_grp - lsp, (i + 1) * ftr_per_grp + lsp, feature_count), j],
                         range(group_count))))
        else:
            input_parts_all = None

        # Scatter data for next prediction and split is among resevoirs for current task
        input_parts = np.empty((ftr_per_grp + 2 * lsp) * res_per_task)
        comm.Scatter(input_parts_all, input_parts, root=master_node_rank)
        input_parts = [input_parts[i * n:(i + 1) * n] for i in range((len(input_parts) + n - 1) // n)]

    # Save results to file on master task
    # print_with_rank('Prediction finished')
    if rank == master_node_rank:
        result_path = work_root + '/results/QGGGG' + dict_to_string(config) + '.txt'
        # print_with_rank('Saving results to ' + result_path)
        np.savetxt(result_path, output)


if __name__ == '__main__':
    main()
