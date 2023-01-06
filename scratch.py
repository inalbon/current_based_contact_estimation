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

result = np.ones((3, 3))

result = np.array(result)
#colors = ['r', 'b', 'g']
colors = ['r']

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel('Amplitude [rad]', labelpad=10)
ax1.set_ylabel('Frequency [Hz]', labelpad=10)
ax1.set_zlabel('MSE')

xlabels = np.array(['0.2', '0.35', '0.5'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['0.1', '0.5', '1.0'])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos=result
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
plt.show()
