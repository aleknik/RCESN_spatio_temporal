import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

predict_length = 1000
train_length = 100000

path = "D:\globus\grid1000-11-fine\RANDOM-QG_approx_res_size-1000_degree-7_feature_count-88_group_count-11_train_length-100000_beta-0.002739326282577479_random_state-42_radius-0.95_lsp-7_predict_length-1000_sigma-0.05.txt"

predicted = np.loadtxt(path)[:, :predict_length]

pd_data = pd.read_csv('../data/QG_everydt_avgu.csv', header=None)

pd_data = pd_data.T
pd_data = (pd_data - pd_data.mean()) / pd_data.std()
pd_data = pd_data.T

all_data = np.asarray(pd_data)

target = all_data[:, train_length:train_length + predict_length]

fig = plt.figure()
ax = plt.axes(xlim=(0, 88), ylim=(-3, 3))
line_predict, = ax.plot([], [], label='predicted')
line_target, = ax.plot([], [], label='target')
ax.legend(loc=1)


def init():
    line_predict.set_data([], [])
    line_target.set_data([], [])
    return [line_predict, line_target]


x = np.arange(target.shape[0])


def animate(i):
    ax.set_title(i + 1)
    line_predict.set_data(x, predicted[:, i])
    line_target.set_data(x, target[:, i])
    return [line_predict, line_target]


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=predict_length, interval=50, blit=True)

anim.save('../anims/anim_1000-11-147.mp4')
