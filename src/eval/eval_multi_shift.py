import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn import decomposition


def calculate_prediction_horizon(errors, horizon):
    for i, val in enumerate(errors):
        if val > horizon:
            return i


def standardize_data(data):
    data = data.T
    data = (data - data.mean()) / data.std()
    return data.T


predict_length = 1000
train_length = 100000

data = pd.read_csv('../../data/QG_everydt_avgu.csv', header=None)

data = standardize_data(data)

data = np.asarray(data)

directory = r"D:\globus\multi_shift"

# PCA
pca = decomposition.PCA(n_components=0.95)
pca.fit(data.T)

l2_error_sum = np.zeros(predict_length)
count = 0
for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    predicted = np.loadtxt(path)[:, :predict_length]
    shift = re.findall(r'shift=\d+', filename)[0].split('=')[1]
    shift = int(shift)

    target = data[:, shift + train_length:shift + train_length + predict_length]

    l2_error_sum += norm(predicted - target, axis=0) / np.mean(norm(target, axis=0))

    count += 1

    predicted = pca.transform(predicted.T).T
    target = pca.transform(target.T).T

    plt.plot(predicted[0, :])
    plt.plot(target[0, :])
    plt.xlabel('t')
    plt.figlegend(['predicted', 'target'], loc='upper left')
    plt.title('Data plot PCA shift=' + str(shift))
    plt.show()

l2_error = l2_error_sum / count

y_horizon = 0.3
x_horizon = calculate_prediction_horizon(l2_error, y_horizon)

plt.plot(l2_error)
plt.xlabel('t')
plt.ylabel('Scaled RMSE')
plt.title('Scaled RMSE plot 0.3 horizon=' + str(x_horizon))
plt.axvline(x=x_horizon, color='r')
plt.axhline(y=y_horizon, color='r')
plt.show()
