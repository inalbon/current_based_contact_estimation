import matplotlib.pyplot as plt
from load_data import LoadData
from feature_engineering import *
from sklearn import linear_model
from scipy import stats
import numpy as np

# load polymander data
data = LoadData()
data.load_polymander_data(dir_name='logs_polymander/one_limb/FL/amp_0.5_freq_0.5', log_name='robot_data_log_2022-12-23_18_44_50')
data.load_polymander_data(dir_name='logs_polymander/one_limb/FL/amp_0.5_freq_0.5', log_name='robot_data_log_2022-12-23_18_47_54')

# load force plate data
data.load_force_plates_data(dir_name='logs_force_plates/one_limb/FL/amp_0.5_freq_0.5', log_name='exp1_amp_0.5_freq_0.5')
data.load_force_plates_data(dir_name='logs_force_plates/one_limb/FL/amp_0.5_freq_0.5', log_name='exp2_amp_0.5_freq_0.5')

# estimation of signal-to-noise ratio of force plate
noise = 2
signal = 6
snr = 20*np.log10(signal/noise)  # dB
print('SNR =', snr)

check = False

for (i, j) in zip(data.list_polymander, data.list_force_plates):
    i.plot_fbck_current(i.t_s, i.fbck_current_data)
    j.plot_forces(j.t_s, j.Fxyz)
    plt.savefig('figures/Fxyz_raw.eps', format='eps')

    # 1) Resampling force plate signal based on polymander signal
    t_s_fp_resampled, Fxyz_resampled = j.resample_force_plate(j.Fxyz, i.t_s)

    # 2) Filter signal
    #i.filtering_signal()
    Fxyz_filtered = filtering_signal(Fxyz_resampled)
    #i.plot_fbck_current()
    j.plot_forces(t_s_fp_resampled, Fxyz_filtered)
    plt.savefig('figures/fbck_current_sigma_6', format='eps')
    plt.savefig('figures/Fxyz_sigma_6.eps', format='eps')

    # 3) Manage delay between polymander and force plate
    t_s_cut, fbck_position_cut, fbck_current_cut, Fxyz_cut = manage_delay_between_poly_and_fp(i.t_s, i.fbck_position_data, i.fbck_current_data, t_s_fp_resampled, Fxyz_filtered, i.frequency)
    plot_aligned_signals(t_s_cut, fbck_position_cut, Fxyz_cut, i.frequency)
    plt.savefig('figures/initial_sequence.eps', format='eps')

    # 4) Remove initial sequence
    t_s_final, fbck_current_final, Fxyz_final = remove_inital_sequence(t_s_cut, fbck_position_cut, fbck_current_cut, Fxyz_cut, i.frequency)
    i.plot_fbck_current(t_s_final, fbck_current_final)
    j.plot_forces(t_s_final, Fxyz_final)

    # 5) Train/test split
    if check is False:
        X_train = fbck_current_final[:, 8:10]
        y_train = Fxyz_final[:, 2]
        check = True
    elif check is True:
        X_test = fbck_current_final[:, 8:10]
        y_test = Fxyz_final[:, 2]


# Machine Learning - multiple linear regression
regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train)
print(model.score(X_train, y_train))

y_pred = regr.predict(X_test)

fig, ax = plt.subplots()
ax.set_title('Multiple Linear Regression')
ax.plot(t_s_final, y_test, label='true value')
ax.plot(t_s_final, y_pred, label='pred')
ax.set(xlabel='time [s]', ylabel='Fz [N]')
ax.legend()

# Machine Learning - linear regression
x = np.absolute(X_train[:, 0])
slope, intercept, r, p, std_err = stats.linregress(x, y_train)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))
fig, ax = plt.subplots()
ax.set_title('Linear Regression')
ax.set(xlabel='Feedback current [mA]', ylabel='Fz [N]')
ax.scatter(x, y_train)
ax.plot(X_train[:, 0], mymodel)

plt.show()
