import matplotlib.pyplot as plt
from load_data import LoadData
from scipy import stats

# load polymander data
data_polymander = LoadData(dir_name='logs_polymander/one_limb/FL/amp_0.35_freq_0.5')
data_polymander.load_polymander_data('robot_data_log_2022-12-23_17_28_23')

# load force plate data
data_force_plate = LoadData(dir_name='logs_force_plates/one_limb/FL/amp_0.35_freq_0.5')
data_force_plate.load_force_plates_data('exp1_amp_0.35_freq_0.5')

poly = data_polymander.list_polymander[0]
force_plate = data_force_plate.list_force_plates[0]

x = poly.fbck_current_data[:, 9]
y = force_plate.Fxyz[:, 2]
print('x', len(x))
print('y', len(y))
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, poly.fbck_current))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
