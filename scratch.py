import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

amplitudes = [0.2, 0.35, 0.5]
frequencies = [0.1, 0.5, 1.0]

list_folder = ['amp_0.2_freq_0.1', 'amp_0.2_freq_0.5', 'amp_0.2_freq_0.5',
               'amp_0.35_freq_0.1', 'amp_0.35_freq_0.5', 'amp_0.35_freq_1.0',
               'amp_0.5_freq_0.1', 'amp_0.5_freq_0.5', 'amp_0.5_freq_1.0']
array_folder = np.array(list_folder).reshape(3, 3)
print(array_folder)


import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Amplitude [rad]', labelpad=10)
ax.set_ylabel('Frequency [Hz]', labelpad=10)
ax.set_zlabel('MSE')

xlabels = np.array(['0.2', '0.35', '0.5'])
ylabels = np.array(['0.1', '0.5', '1.0'])

x, y = np.random.rand(2, 100) * 3
hist, xedges, yedges = np.histogram2d(x, y, bins=3, range=[[0, 3], [0, 3]])

# Construct arrays for the anchor positions of the 9 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")

xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0


# Construct arrays with the dimensions for the 9 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Set ticks
ax.w_xaxis.set_ticks(ypos[0:3] + dx/2.)
ax.w_xaxis.set_ticklabels(xlabels)

ax.w_yaxis.set_ticks(ypos[0:3] + dy/2.)
ax.w_yaxis.set_ticklabels(ylabels)

# Set colors
values = np.linspace(0.2, 1., xpos.ravel().shape[0])
colors = cm.rainbow(values)

# Plot 3D bars
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.6, color=colors)
print(xpos, ypos, zpos)
print(dx, dy, dz)

# Plot error bars
x_error = [[x + dx]*3 for x in range(3)]
x_error = [item for sublist in x_error for item in sublist]
y_error = [(y + dy) % 3 for y in range(9)]

for (i, j, k) in zip(x_error, y_error, dz):
    ax.errorbar(i, j, k, 3, color='black', capsize=4)

plt.show()

# result = np.ones((3, 3))
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlabel('Amplitude [rad]', labelpad=10)
# ax.set_ylabel('Frequency [Hz]', labelpad=10)
# ax.set_zlabel('MSE')
#
# xlabels = np.array(['0.2', '0.35', '0.5'])
# xpos = np.arange(xlabels.shape[0])
# print(xpos)
# ylabels = np.array(['0.1', '0.5', '1.0'])
# ypos = np.arange(ylabels.shape[0])
# print((ypos))
#
# xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
#
# zpos = result
# zpos = zpos.ravel()
#
# dx = 0.5
# dy = 0.5
# dz = zpos
#
# ax.w_xaxis.set_ticks(xpos + dx/2.)
# ax.w_xaxis.set_ticklabels(xlabels)
#
# ax.w_yaxis.set_ticks(ypos + dy/2.)
# ax.w_yaxis.set_ticklabels(ylabels)
#
# values = np.linspace(0.2, 1., xposM.ravel().shape[0])
# colors = cm.rainbow(values)
#
# ax.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
# plt.show()
