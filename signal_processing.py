import matplotlib.pyplot as plt
from load_data import LoadData
from sklearn import linear_model
from scipy import signal
from scipy import stats
import numpy as np

# load polymander data
data_polymander = LoadData(dir_name='logs_polymander/one_limb/FL/amp_0.5_freq_0.5')
data_polymander.load_polymander_data('robot_data_log_2022-12-23_18_44_50')  # train
data_polymander.load_polymander_data('robot_data_log_2022-12-23_18_47_54')  # test

# load force plate data
data_force_plate = LoadData(dir_name='logs_force_plates/one_limb/FL/amp_0.5_freq_0.5')
data_force_plate.load_force_plates_data('exp1_amp_0.5_freq_0.5')  # train
data_force_plate.load_force_plates_data('exp2_amp_0.5_freq_0.5')  # test

# estimation of signal-to-noise ratio of force plate
noise = 2
signal = 6
snr = 20*np.log10(signal/noise)  # dB
print('SNR =', snr)

for (i, j) in zip(data_polymander.list_polymander, data_force_plate.list_force_plates):
    j.plot_log()
    plt.savefig('figures/Fxyz_raw.eps', format='eps')
    # 1) Resampling force plate signal based on polymander signal
    j.resample(i.t_s)

    # 2) Filter force plate signal
    j.filtering_signal()
    j.plot_log()
    plt.savefig('figures/Fxyz_sigma_6.eps', format='eps')

    # 3) Manage delay
    # 3.1) Detect initial sequence (4 steps)
    # Find minima in roll motor (when the limb touch the force plate)
    minima = i.detect_initial_sequence(True)
    # Find peaks in force plate (when the limb touch the force plate)
    peaks = j.detect_initial_sequence(True)

    offset = []
    for k in range(4):
        offset.append(peaks[k]-minima[k])
    offset = np.mean(offset)

    # 3.2) Cut signal to ensure that both signals are aligned according to the peaks
    j.cut_signal(offset, offset+len(i.t_s))  # remove first seconds in force plate signal
    i.cut_signal(0, len(j.t_s))  # remove extra seconds in polymander
    print(i.t_s[0], j.t_s[0], i.t_s[-1], j.t_s[-1])
    j.plot_log()
    minima = i.detect_initial_sequence(True)
    fig, ax = plt.subplots()
    ax.set_title(j.log_name)
    ax.set(xlabel='time [s]')
    ax.plot(i.t_s, j.Fxyz[:, 2], label='Fz [N]')
    ax.plot(i.t_s, i.fbck_position_data[:, 9], label='fbck position of limb [rad]')
    ax.vlines(i.t_s[minima[0:4]], 0, max(j.Fxyz[:, 2]), colors='lime', linestyles='dashed', label='first 4 steps')
    ax.legend(loc='upper right')
    plt.savefig('figures/initial_sequence.eps', format='eps')

    # 3.3) Remove initial sequence
    i.cut_signal(minima[4], minima[-1])
    j.cut_signal(minima[4], minima[-1])
    i.plot_fbck_current()
    j.plot_log()

# training
X = data_polymander.list_polymander[0].fbck_current_data[:, 8:10]
y = data_force_plate.list_force_plates[0].Fxyz[:, 2]

# prediction
X_pred = data_polymander.list_polymander[1].fbck_current_data[:, 8:10]

# true value
y_test = data_force_plate.list_force_plates[1].Fxyz[:, 2]

# Machine Learning - multiple regression
regr = linear_model.LinearRegression()
model = regr.fit(X, y.reshape(-1, 1))
print(model.score(X, y.reshape(-1, 1)))

y_pred = regr.predict(X_pred)

plt.figure()
plt.title('Fz prediction')
plt.plot(data_polymander.list_polymander[1].t_s, y_test, label='true value')
plt.plot(data_polymander.list_polymander[1].t_s, y_pred, label='pred')
plt.legend()

plt.show()
