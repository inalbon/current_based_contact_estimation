import matplotlib.pyplot as plt
from load_data import LoadData
import numpy as np


# load polymander data
data_polymander = LoadData(dir_name='logs_polymander/one_limb/FL/amp_0.2_freq_0.1')
data_polymander.load_polymander_data()

# load force plate data
data_force_plate = LoadData(dir_name='logs_force_plates/one_limb/FL/amp_0.2_freq_0.1')
data_force_plate.load_force_plates_data()

data = data_polymander.list_polymander[0]
data2 = data_force_plate.list_force_plates[0]

data.plot_goal_position()
data.plot_fbck_position()
data.plot_fbck_current()
# data.plot_fbck_voltage()

data2.plot_log()

plt.show()
