import matplotlib.pyplot as plt
from load_data import LoadData
import numpy as np


# load polymander data
data_polymander = LoadData(dir_name='logs_polymander/FL')
data_polymander.load_polymander_data(log_name='robot_data_log_2022-11-03_10_29_33')

# load force plates data
data_force_plates = LoadData(dir_name='logs_force_plates/polymander/FL')
data_force_plates.load_force_plates_data(log_name='exp1')

data = data_polymander.list_polymander[0]



data.plot_goal_position()
data.plot_fbck_position()
data.plot_fbck_current()
data.plot_fbck_voltage()

plt.show()
