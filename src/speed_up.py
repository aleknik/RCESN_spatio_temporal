import matplotlib.pyplot as plt
import numpy as np

y = np.array([72252.91609, 36022.8004181, 19718.9969389, 9850.34015107, 4847.59368491, 2521.27283096, 1487.54309106])
x = np.array([1, 2, 4, 8, 16, 32, 64])

# plt.subplot(2, 1, 1)
# plt.plot(x, y)
# plt.ylabel('time (s)')
# plt.xlabel('number of parallel tasks')
#
# plt.subplot(2, 1, 2)
# plt.plot(x, y[0] / y)
# plt.ylabel('S/P')72252.91609
# plt.xlabel('number of parallel tasks')
#
# plt.show()


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('number of parallel tasks')
ax1.set_ylabel('time (s)', color=color)
ax1.plot(x, y, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Ts/Tp', color=color)  # we already handled the x-label with ax1
ax2.plot(x, y[0] / y, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
