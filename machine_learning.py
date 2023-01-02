import matplotlib.pyplot as plt
import numpy as np
from load_data import LoadData
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
poly = LoadData(dir_name='logs_polymander/one_limb/FL/amp_0.5_freq_0.5', data_type='poly')
force_plate = LoadData(dir_name='logs_force_plates/one_limb/FL/amp_0.5_freq_0.5', data_type='force_plate')

list_X = poly.list_polymander
list_y = force_plate.list_force_plates
print(list_X[0].fbck_current_headers)
print(list_y[0].headers)

features = list_X[0].fbck_current_headers
target = list_y[0].headers[3]

# Features engineering
check = False
for (i, j) in zip(list_X, list_y):
    i.plot_fbck_current()
    j.plot_log()
    plt.savefig('figures/Fxyz_raw.eps', format='eps')
    # 1) Resampling force plate signal based on polymander signal
    j.resample(i.t_s)

    # 2) Filter signal
    # i.filtering_signal()
    j.filtering_signal()
    # i.plot_fbck_current()
    j.plot_log()
    plt.savefig('figures/fbck_current_sigma_6', format='eps')
    plt.savefig('figures/Fxyz_sigma_6.eps', format='eps')

    # 3) Manage delay
    # 3.1) Detect initial sequence (4 steps)
    # Find minima in roll motor (when the limb touch the force plate)
    minima = i.detect_initial_sequence(False)
    # Find peaks in force plate (when the limb touch the force plate)
    peaks = j.detect_initial_sequence(False)

    offset = []
    for k in range(4):
      offset.append(peaks[k] - minima[k])
    offset = np.mean(offset)

    # 3.2) Cut signal to ensure that both signals are aligned according to the peaks
    j.cut_signal(offset, offset + len(i.t_s))  # remove first seconds in force plate signal
    i.cut_signal(0, len(j.t_s))  # remove extra seconds in polymander
    j.t_s = i.t_s  # adjust force plate time on polymander time

    minima = i.detect_initial_sequence(False)
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

    if check is False:
        X = i.fbck_current_data
        y = j.Fxyz[:, 2]
        check = True
    elif check is True:
        print(np.shape(i.fbck_current_data), np.shape(X))
        X = np.concatenate((X, i.fbck_current_data))
        y = np.concatenate((y, j.Fxyz[:, 2]))

print(np.shape(X))
print(np.shape(y))
plt.figure()
plt.plot(X[:, 8:10])
plt.figure()
plt.plot(y)
# Machine learning - linear regression
# Train and test ratio 0.75
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)
# regr = LinearRegression()
# regr.fit(X_train, y_train)
# regr.predict(X_test)

plt.show()



