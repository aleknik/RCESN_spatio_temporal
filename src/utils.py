import os

import numpy as np
import pandas as pd

import arg_parser


def dict_to_string(dict):
    result = ''
    for index, item in enumerate(sorted(dict.items())):
        key, val = item
        if index != 0:
            result += '-'
        result += str(key) + '=' + str(val)

    return result


def standardize_data(data):
    data = data.T
    data = (data - data.mean()) / data.std()
    return data.T


def load_data(work_root, path, transpose=False):
    pd_data = pd.read_csv(os.path.join(work_root, path), header=None)
    if transpose:
        pd_data = pd_data.T
    pd_data = standardize_data(pd_data)
    return np.array(pd_data)


def get_config(config):
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
