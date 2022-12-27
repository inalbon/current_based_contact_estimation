import matplotlib.pyplot as plt
from load_data import LoadData
import numpy as np


# load polymander data
data_polymander = LoadData(dir_name='logs_polymander/four_limbs/FL')
data_polymander.load_polymander_data()

# load force plates data
data_force_plates = LoadData(dir_name='logs_force_plates/four_limbs/FL')
data_force_plates.load_force_plates_data()

# Signal processing
data_force_plates = data_force_plates.list_force_plates[0]
data_polymander = data_polymander.list_polymander[0]
frequency_poly = len(data_polymander.t_s)/data_polymander.t_s[-1]
frequency_force_plate = len(data_force_plates.t_s)/data_force_plates.t_s[-1]

plt.figure()
data_force_plates.plot_log()

# filtering signal from force plate
data_force_plates.filtering_signal()
data_force_plates.plot_log()

# manage delay between force plate and polymander
start_recording_force_plate = 10*60*60+29*60+50 - 15
start_recording_poly = 10*60*60+29*60+33
delay = abs(start_recording_force_plate - start_recording_poly)
poly_time_recording = data_polymander.t_s[-1]

data_force_plates.cut_signal(delay, poly_time_recording)
plt.figure()
data_force_plates.plot_log()

# currents not downsampled
data_polymander.plot_fbck_current()

print('shape of time data and currents data before downsampling', np.shape(data_polymander.t_s), np.shape(data_polymander.fbck_current_data))
print('recording time in [s]', data_polymander.t_s[-1]-data_polymander.t_s[0])

print('shape of time and force plate', np.shape(data_force_plates.t_s), np.shape(data_force_plates.Fxyz))
print('recording time in [s]', data_force_plates.t_s[-1]-data_force_plates.t_s[0])
print(f'f_poly = {frequency_poly}, f_force_plate = {frequency_force_plate}')

# currents downsampled
data_polymander.downsampling(np.shape(data_force_plates.t_s)[0])
print('shape of currents data before downsampling', np.shape(data_polymander.fbck_current_data), np.shape(data_polymander.t_s))

frequency_poly = len(data_polymander.t_s)/data_polymander.t_s[-1]
frequency_force_plate = len(data_force_plates.t_s)/data_force_plates.t_s[-1]
print(f'f_poly = {frequency_poly}, f_force_plate = {frequency_force_plate}')

plt.figure()
plt.plot(data_polymander.fbck_current_data[:, 9], data_force_plates.Fxyz[:, 2])

data_polymander.plot_fbck_current()

plt.show()
