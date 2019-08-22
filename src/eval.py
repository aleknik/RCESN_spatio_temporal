import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm


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

# predict_length = train_length + predict_length
# train_length = 0

# path = '../results/QG_beta-0.2_degree-6_number_of_features-88_number_of_reservoirs-11_overlap_size-11_prediction_size-2000_radius-0.7_reservoir_size-5000_sigma-0.005_training_size-100000.txt'
# path = '../comp/RANDOM-QG_approx_res_size-2000_degree-6_feature_count-88_group_count-11_train_length-100000_beta-4.2975297258771334_random_state-42_radius-0.24380000000000002_lsp-12_predict_length-2000_sigma-0.04561224489795918.txt'
# path = 'D:\globus\grid2000\RANDOM-QG_approx_res_size-2000_degree-9_feature_count-88_group_count-11_train_length-100000_beta-0.5797897897897898_random_state-42_radius-0.9420999999999999_lsp-15_predict_length-1000_sigma-0.06075775775775776.txt'
# path = '../results/NoBias_QG_beta-0.6_degree-9_number_of_features-88_number_of_reservoirs-1_overlap_size-0_prediction_size-1000_radius-0.95_reservoir_size-1000_sigma-0.5_training_size-100000.txt'
# path = "D:\globus\QG_beta-0.6_degree-9_number_of_features-88_number_of_reservoirs-11_overlap_size-15_prediction_size-2000_radius-0.95_reservoir_size-10000_sigma-0.06_training_size-100000.txt"
path = r"D:\globus\grid5000-11\RANDOM-QG_approx_res_size-5000_degree-9_feature_count-88_group_count-11_train_length-100000_beta-0.02250273073630341_random_state-42_radius-0.95_lsp-5_predict_length-1000_sigma-0.05.txt"
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

target = np.asarray(data)[:, shift + train_length:shift + train_length + predict_length]
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
print(calculate_prediction_horizon(l2_error, 0.3))
plt.plot(l2_error)
plt.show()

graph_count = 4
offset = 0
for i in range(graph_count):
    plt.subplot(graph_count / 2, 2, i + 1)
    plt.plot(x, predicted[i + offset, :])
    plt.plot(x, target[i + offset, :])
    plt.title("Variable " + str(i + 1 + offset))
    axes = plt.gca()
    axes.set_ylim([-3, 3])

plt.figlegend(['predicted', 'target'], loc='upper left')
plt.show()
