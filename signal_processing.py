from load_data import LoadData
from feature_engineering import *
import matplotlib.pyplot as plt
from sklearn import linear_model

# load data
data = LoadData()
data.load_polymander_data(dir_name='logs_polymander/one_limb/FL/amp_0.35_freq_0.5', log_name='robot_data_log_2022-12-23_17_28_23')
data.load_force_plates_data(dir_name='logs_force_plates/one_limb/FL/amp_0.35_freq_0.5', log_name='exp1_amp_0.35_freq_0.5')

i = data.list_polymander[0]
j = data.list_force_plates[0]

i.plot_fbck_current(i.t_s, i.fbck_current_data)
j.plot_forces(j.t_s, j.Fxyz)

# 1) Filter signal
fbck_current_filtered = filtering_signal(i.fbck_current_data, 10)
Fxyz_filtered = filtering_signal(j.Fxyz, 20)

i.plot_fbck_position(i.t_s[0:4000], i.fbck_position_data[0:4000, :])
i.plot_fbck_current(i.t_s, fbck_current_filtered)
j.plot_forces(j.t_s, Fxyz_filtered)

indices = range(2000, 4000)
fig, ax = plt.subplots()
ax.set_title(j.log_name)
ax.set(xlabel='time [s]', ylabel='Force [N]')
ax.plot(j.t_s[indices], j.Fxyz[indices, 2], label='Fz raw')
ax.plot(j.t_s[indices], filtering_signal(j.Fxyz, 10)[indices, 2], label='Fz filtered, $\sigma = 10$')
ax.plot(j.t_s[indices], filtering_signal(j.Fxyz, 20)[indices, 2], 'cyan', label='Fz filtered, $\sigma = 20$')
ax.legend()
plt.savefig('figures/signal_processing/Fz_filtered.eps', format='eps')

fig, ax = plt.subplots()
ax.set_title(i.log_name)
ax.set(xlabel='time [s]', ylabel='Fbck Current [mA]')
ax.plot(i.t_s[indices], i.fbck_current_data[indices, 9], label=f'{i.fbck_current_headers[9][0:12]} raw')
ax.plot(i.t_s[indices], filtering_signal(i.fbck_current_data[indices, 9], 10), label=f'{i.fbck_current_headers[9][0:12]} filtered, $\sigma = 10$')
ax.plot(i.t_s[indices], filtering_signal(i.fbck_current_data[indices, 9], 20), 'cyan', label=f'{i.fbck_current_headers[9][0:12]} filtered, $\sigma = 20$')
ax.legend()
plt.savefig('figures/signal_processing/current9_filtered.eps', format='eps')


# 2) Resampling force plate signal based on polymander signal
t_s_fp_resampled, Fxyz_resampled = j.resample_force_plate(Fxyz_filtered, i.t_s)

# 3) Manage delay between polymander and force plate
t_s_cut, fbck_position_cut, fbck_current_cut, Fxyz_cut = manage_delay_between_poly_and_fp(i.t_s, i.fbck_position_data,
                                                                                          fbck_current_filtered,
                                                                                          t_s_fp_resampled,
                                                                                          Fxyz_resampled, i.frequency)

plot_aligned_signals(t_s_cut, fbck_position_cut, Fxyz_cut, i.frequency)
plt.savefig('figures/signal_processing/initial_sequence.eps', format='eps')

# 4) Remove initial sequence
t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = remove_inital_sequence(t_s_cut, fbck_position_cut,
                                                                                        fbck_current_cut, Fxyz_cut,
                                                                                        i.frequency)

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(t_s_final, fbck_current_final[:, 8:10], label=i.fbck_current_headers[8:10])
ax[0].set(xlabel='time [s]', ylabel='Feedback current [mA]')
ax[1].plot(t_s_final, Fxyz_final[:, 2], label=j.headers[3])
ax[1].set(xlabel='time [s]', ylabel='Force [N]')
for ax in ax.flat:
    ax.legend()
plt.savefig('figures/signal_processing/crop_data.eps', format='eps')


X = fbck_current_final[:, 8:10]
Y = Fxyz_final[:, 2]

plt.show()
