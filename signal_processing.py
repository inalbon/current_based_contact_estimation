import matplotlib.pyplot as plt
from load_data import LoadData
from sklearn import linear_model
from scipy import signal
import numpy as np

# load polymander data
data_polymander = LoadData(dir_name='logs_polymander/one_limb/FL/amp_0.35_freq_0.5')
data_polymander.load_polymander_data('robot_data_log_2022-12-23_17_28_23')  # train
data_polymander.load_polymander_data('robot_data_log_2022-12-23_17_33_24')  # test

# load force plate data
data_force_plate = LoadData(dir_name='logs_force_plates/one_limb/FL/amp_0.35_freq_0.5')
data_force_plate.load_force_plates_data('exp1_amp_0.35_freq_0.5')  # train
data_force_plate.load_force_plates_data('exp2_amp_0.35_freq_0.5')  # test

for (i, j) in zip(data_polymander.list_polymander, data_force_plate.list_force_plates):
    j.plot_log()
    # 1) Resampling force plate signal based on polymander signal
    j.resample(i.t_s)
    j.plot_log()

    #print('test', i.t_s[340:346], i.t_s[len(j.t_s)])
    #print('test', j.t_s[340:346], j.t_s[-1])

    # 2) Filter force plate signal
    j.filtering_signal()
    j.plot_log()

    # 3) Manage delay
    # Detect initial sequence (4 steps on the force plate)
    # Find peaks in force plate
    peaks, _ = signal.find_peaks(j.Fxyz[:, 2], prominence=1, width=100)  # or height=4, distance=200
    plt.figure()
    plt.plot(j.Fxyz[:, 2], label=j.headers[2])
    plt.plot(peaks, j.Fxyz[peaks, 2], "x")
    #print('Fz peaks =', peaks[0:4])

    # Find minima in roll motor (when the limb touch the force plate)
    minima, _ = signal.find_peaks(-i.fbck_position_data[:, 9], prominence=-0.5, width=100, distance=100)
    plt.plot(i.fbck_position_data[:, 9], label=i.fbck_position_headers[9])
    plt.plot(minima, i.fbck_position_data[minima, 9], "x")
    plt.legend()
    #print('Fbck pos. minima =', minima[0:4])

    offset = 0
    for k in range(4):
        offset += (peaks[k]-minima[k])/4
    print('Average delay in steps =', offset)

    # Cut signal
    j.cut_signal(offset, len(i.fbck_position_data))
    j.plot_log()
    plt.figure()
    plt.plot(j.Fxyz[:, 2], label=j.headers[2])
    plt.plot(i.fbck_position_data[:, 9], label=i.fbck_position_headers[9])
    plt.vlines(peaks, 0, max(i.fbck_position_data[:, 9]), label=['1', '2', '3', '4'])
    plt.legend()
    print(len(i.fbck_position_data[:, 9]), len(j.Fxyz[:, 2]))



# # training
# X = data_polymander.list_polymander[0].fbck_current_data[:, 8:10]
# y = data_force_plate.list_force_plates[0].Fxyz[:, 2]
#
# # prediction
# X_pred = data_polymander.list_polymander[1].fbck_current_data[:, 8:10]
#
# # true value
# y_test = data_force_plate.list_force_plates[1].Fxyz[:, 2]
#
# # Machine Learning - multiple regression
# regr = linear_model.LinearRegression()
# model = regr.fit(X, y.reshape(-1, 1))
# print(model.score(X, y.reshape(-1, 1)))
#
# y_pred = regr.predict(X_pred)
#
# plt.figure()
# plt.plot(data_polymander.list_polymander[1].t_s, y_test, label='true value')
# plt.plot(data_polymander.list_polymander[1].t_s, y_pred, label='pred')
# plt.legend()

plt.show()
