import os
import re

import numpy as np
import pandas as pd
from numpy.linalg import norm


def calculate_prediction_horizon(errors, horizon):
    for i, val in enumerate(errors):
        if val > horizon:
            return i

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


predict_length = 1000
train_length = 100000
shift = 0

pd_data = pd.read_csv('../../data/QG_everydt_avgu.csv', header=None)

pd_data = pd_data.T
pd_data = (pd_data - pd_data.mean()) / pd_data.std()
pd_data = pd_data.T

all_data = np.asarray(pd_data)

target = all_data[:, shift + train_length:shift + train_length + predict_length]

directory = r"D:\stampede\random_shift_results"

target_norm_mean = np.mean(norm(target, axis=0))

res = {}
prediction_horizon = 0.3

for dir_name in get_immediate_subdirectories(directory):
    l2_error_sum = np.zeros(predict_length)
    count = 0
    for filename in os.listdir(os.path.join(directory, dir_name)):
        path = os.path.join(directory, dir_name, filename)
        predicted = np.loadtxt(path)[:, :predict_length]
        shift = re.findall(r'shift-\d+', filename)[0].split('-')[1]
        shift = int(shift)

        target = all_data[:, shift + train_length:shift + train_length + predict_length]

        l2_error_sum += norm(predicted - target, axis=0) / np.mean(norm(target, axis=0))

        count += 1

    l2_error = l2_error_sum / count

    res[dir_name] = calculate_prediction_horizon(l2_error, prediction_horizon)

s = [(k, res[k]) for k in sorted(res, key=res.get, reverse=False)]
for k, v in s:
    print(v, k)
