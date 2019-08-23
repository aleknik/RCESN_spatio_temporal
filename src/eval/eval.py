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

shift = 0

path = r"D:\globus\multi_shift_test\data=QG-beta=0.004-degree=7-number_of_features=88-number_of_reservoirs=11-overlap_size=7-prediction_size=1000-radius=0.95-reservoir_size=1000-shift=0-sigma=0.05-training_size=100000.txt"
predicted = np.loadtxt(path)[:, :predict_length]

le = 1
step = 1

x = np.arange(predict_length) * step * le

# predicted = np.loadtxt('results/QG_Nodes_11_ResCount_22_L_6_train_10000_res_100.txt')

# pd_data = pd.read_csv('data/3tier_lorenz_v3.csv', header=None).T
# pd_data = pd.read_csv('data/ks_512.csv', header=None)
# pd_data = pd.read_csv('data/KS_data.csv', header=None)
# pd_data = pd.read_csv('data/ks_64.csv', header=None)

data = pd.read_csv('../../data/QG_everydt_avgu.csv', header=None)

data = standardize_data(data)

data = np.asarray(data)

target = data[:, shift + train_length:shift + train_length + predict_length]

data_min = target.min()
data_max = target.max()

fig, axes = plt.subplots(nrows=3, ncols=1)
ax1, ax2, ax3 = axes
im1 = ax1.imshow(target, vmin=data_min, vmax=data_max, cmap='jet', aspect='auto',
                 extent=[x[0], x[-1], target.shape[0], 0])
ax1.set_title('Target')
im2 = ax2.imshow(predicted, vmin=data_min, vmax=data_max, cmap='jet', aspect='auto',
                 extent=[x[0], x[-1], target.shape[0], 0])
ax2.set_title('Predicted')
im3 = ax3.imshow(target - predicted, cmap='jet', aspect='auto',
                 extent=[x[0], x[-1], target.shape[0], 0])
ax3.set_title('Prediction error')

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
fig.colorbar(im3, ax=ax3)

plt.show()

l2_error = norm(predicted - target, axis=0) / np.mean(norm(target, axis=0))
print(calculate_prediction_horizon(l2_error, 0.3))
plt.plot(l2_error)
plt.xlabel('t')
plt.ylabel('RMSE')
plt.title('Error plot')
plt.show()

# PCA
pca = decomposition.PCA(n_components=1)
pca.fit(data.T)
predicted = pca.transform(predicted.T).T
target = pca.transform(target.T).T

plt.plot(x, predicted[0, :])
plt.plot(x, target[0, :])
plt.xlabel('t')
plt.figlegend(['predicted', 'target'], loc='upper left')
plt.title('Data plot PCA')
plt.show()

l2_error = norm(predicted - target, axis=0) / np.mean(norm(target, axis=0))
print(calculate_prediction_horizon(l2_error, 0.3))
plt.plot(l2_error)
plt.xlabel('t')
plt.ylabel('RMSE')
plt.title('Error plot PCA')
plt.show()

# graph_count = 4
# offset = 0
# for i in range(graph_count):
#     plt.subplot(graph_count / 2, 2, i + 1)
#     plt.plot(x, predicted[i + offset, :])
#     plt.plot(x, target[i + offset, :])
#     plt.title("Variable " + str(i + 1 + offset))
#     axes = plt.gca()
#     axes.set_ylim([-3, 3])
#
# plt.figlegend(['predicted', 'target'], loc='upper left')
# plt.show()
