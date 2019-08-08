import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

predict_length = 200
train_length = 50000

path = '../comp/QG_number_of_reservoirs-1_number_of_features-88_reservoir_size-1000_training_size-50000_prediction_size-1000_overlap_size-0_sigma-0.5_radius-0.9_beta-0.01.txt'

predicted = np.loadtxt(path)[:, :predict_length]

pd_data = pd.read_csv('../data/QG_everydt_avgu.csv', header=None)

pd_data = pd_data.T
pd_data = (pd_data - pd_data.mean()) / pd_data.std()
pd_data = pd_data.T

target = np.asarray(pd_data)[:, train_length:train_length + predict_length]

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

anim.save('../anims/anim_0.01.mp4')
