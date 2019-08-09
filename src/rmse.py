import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm

predict_length = 1000
train_length = 100000

pd_data = pd.read_csv('../data/QG_everydt_avgu.csv', header=None)

pd_data = pd_data.T
pd_data = (pd_data - pd_data.mean()) / pd_data.std()
pd_data = pd_data.T

target = np.asarray(pd_data)[:, train_length:train_length + predict_length]
# predicted = target

# target = np.load('data/Expansion_2step_backR_size_train_5000_Rd_0.1_Shift_0.npy')

directory = '../comp'

target_norm_mean = np.mean(norm(target, axis=0))

for filename in os.listdir(directory):
    num_res = re.findall(r'number_of_reservoirs-\d+', filename)[0].split('-')[1]
    num_l = re.findall(r'overlap_size-\d+', filename)[0].split('-')[1]
    res_size = re.findall(r'reservoir_size-\d+', filename)[0].split('-')[1]
    sigma = re.findall(r'sigma-\d*\.?\d*', filename)[0].split('-')[1]
    radius = re.findall(r'radius-\d*\.?\d*', filename)[0].split('-')[1]
    # beta = re.findall(r'beta-\d*\.?\d*', filename)[0].split('-')[1]
    beta = 0.0001
    # degree = re.findall(r'degree-\d*\.?\d*', filename)[0].split('-')[1]
    degree = 0.0001
    path = os.path.join(directory, filename)
    predicted = np.loadtxt(path)[:, :predict_length]

    l2_error = norm(predicted - target, axis=0) / target_norm_mean
    plt.plot(l2_error,
             label='res=%s size=%s l=%s s=%s rad=%s b=%s d=%s' % (num_res, res_size, num_l, sigma, radius, beta, degree))
    # plt.plot(mse, label=filename[-60:])

plt.legend()
plt.show()
