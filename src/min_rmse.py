import os

import numpy as np
import pandas as pd
from numpy.linalg import norm


def calculate_prediction_horizon(errors, horizon):
    for i, val in enumerate(errors):
        if val > horizon:
            return i


predict_length = 1000
train_length = 100000
shift = 0

pd_data = pd.read_csv('../data/QG_everydt_avgu.csv', header=None)

pd_data = pd_data.T
pd_data = (pd_data - pd_data.mean()) / pd_data.std()
pd_data = pd_data.T

all_data = np.asarray(pd_data)

target = all_data[:, shift + train_length:shift + train_length + predict_length]

directory = "D:\globus\grid2000-11-fine"

target_norm_mean = np.mean(norm(target, axis=0))

res = {}
prediction_horizon = 0.3
for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    predicted = np.loadtxt(path)[:, :predict_length]

    l2_error = norm(predicted - target, axis=0) / target_norm_mean

    res[filename] = calculate_prediction_horizon(l2_error, prediction_horizon)

s = [(k, res[k]) for k in sorted(res, key=res.get, reverse=False)]
for k, v in s:
    print(v, k)
