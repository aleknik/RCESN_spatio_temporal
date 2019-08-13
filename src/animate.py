import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

predict_length = 500
train_length = 100000

path = "D:\globus\grid2000\RANDOM-QG_approx_res_size-2000_degree-9_feature_count-88_group_count-11_train_length-100000_beta-0.5797897897897898_random_state-42_radius-0.9420999999999999_lsp-15_predict_length-1000_sigma-0.06075775775775776.txt"

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
                     frames=predict_length, interval=10, blit=True)

anim.save('../anims/anim_2000-horizon.mp4')
