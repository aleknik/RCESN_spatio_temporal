import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm


def standardize_data(data):
    data = data.T
    data = (data - data.mean()) / data.std()
    return data.T


predict_length = 200
train_length = 100000

# predict_length = train_length + predict_length
# train_length = 0

path = '../comp/QG_beta-100.0_degree-3_number_of_features-88_number_of_reservoirs-11_overlap_size-6_prediction_size-2000_radius-0.99_reservoir_size-1000_sigma-0.5_training_size-100000.txt'

predicted = np.loadtxt(path)[:, :predict_length]

le = 1
step = 1

x = np.arange(predict_length) * step * le

# predicted = np.loadtxt('results/QG_Nodes_11_ResCount_22_L_6_train_10000_res_100.txt')

# pd_data = pd.read_csv('data/3tier_lorenz_v3.csv', header=None).T
# pd_data = pd.read_csv('data/ks_512.csv', header=None)
# pd_data = pd.read_csv('data/KS_data.csv', header=None)
# pd_data = pd.read_csv('data/ks_64.csv', header=None)

data = pd.read_csv('../data/QG_everydt_avgu.csv', header=None)

data = standardize_data(data)

target = np.asarray(data)[:, train_length:train_length + predict_length]
# predicted = target

# target = np.load('data/Expansion_2step_backR_size_train_5000_Rd_0.1_Shift_0.npy')

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
im3 = ax3.imshow(target - predicted, vmin=data_min, vmax=data_max, cmap='jet', aspect='auto',
                 extent=[x[0], x[-1], target.shape[0], 0])
ax3.set_title('Prediction error')

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
fig.colorbar(im3, ax=ax3)

plt.show()

l2_error = norm(predicted - target, axis=0) / np.mean(norm(target, axis=0))
plt.plot(l2_error)
plt.show()

graph_count = 4
for i in range(graph_count):
    plt.subplot(graph_count / 2, 2, i + 1)
    plt.plot(x, predicted[i, :])
    plt.plot(x, target[i, :])
    plt.title("Variable " + str(i + 1))
    axes = plt.gca()
    axes.set_ylim([-3, 3])

plt.figlegend(['predicted', 'target'], loc='upper left')
plt.show()
