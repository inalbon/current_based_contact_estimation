import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import numpy as np


def signal_processing(poly, force_plate):  # prepare data for training
    # 1) Filter signal
    fbck_current_filtered = filtering_signal(poly.fbck_current_data, 10)
    Fxyz_filtered = filtering_signal(force_plate.Fxyz, 20)

    # 2) Resampling force plate signal based on polymander signal
    t_s_fp_resampled, Fxyz_resampled = resample_signal(Fxyz_filtered, force_plate.t_s, poly.t_s)

    # 3) Manage delay
    t_s_cut, fbck_position_cut, fbck_current_cut, Fxyz_cut = manage_delay_between_poly_and_fp(poly.t_s,
                                                                                              poly.fbck_position_data,
                                                                                              fbck_current_filtered,
                                                                                              t_s_fp_resampled,
                                                                                              Fxyz_resampled,
                                                                                              poly.frequency)

    # 4) Remove initial sequence
    t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = remove_initial_sequence(t_s_cut, fbck_position_cut,
                                                                                            fbck_current_cut,
                                                                                            Fxyz_cut, poly.frequency)
    return t_s_final, fbck_position_final, fbck_current_final, Fxyz_final


def filtering_signal(signal, sigma=6):
    filtered_signal = gaussian_filter1d(signal, sigma=sigma, axis=0)
    return filtered_signal


def resample_signal(signal_to_resample, time_to_resample, time_wanted):
    print(f'Recording time of robot: [{time_wanted[0]}, {time_wanted[-1]}]')
    print(f'Recording time of force plate: [{time_to_resample[0]}, {time_to_resample[-1]}]')
    print(f'sizes {np.shape(signal_to_resample)}, {np.shape(time_to_resample)}, {np.shape(time_wanted)}')

    # Find number of samples in robot data and convert to corresponding number of samples wanted in force plate
    nb_steps = int(len(time_wanted)*time_to_resample[-1]/time_wanted[-1])
    print(f'size before {len(time_to_resample)}, after {nb_steps}')

    # Resampling
    signal_resampled, t_s_resampled = signal.resample(signal_to_resample, nb_steps, time_to_resample)
    print(f'sizes resampled {np.shape(t_s_resampled)}, {np.shape(signal_resampled)}')
    print(t_s_resampled[-1])
    return t_s_resampled, signal_resampled


def cut_time(t_s, start, end):
    t_s_cut = t_s[int(start):int(end)]
    return t_s_cut


def cut_signal(init_signal, start, end):
    signal_cut = init_signal[int(start):int(end), :]
    return signal_cut


def detect_minima_of_Fz(fbck_position, frequency):
    # Find minimum in feedback position -> when the limb touches the ground
    ratio = 0.5 / frequency  # width and distance of peaks are influenced by the frequency
    minima, _ = signal.find_peaks(-fbck_position[:, 9], prominence=-0.5, width=100 * ratio, distance=150 * ratio)
    return minima


def detect_initial_sequence_polymander(fbck_position, frequency, plot=False):
    # Detect initial sequence (4 steps on the force plate)
    # Find minimum in feedback position -> when the limb touches the ground
    minima = detect_minima_of_Fz(fbck_position, frequency)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fbck_position[:, 9], label='9FbckPosition')
        ax.plot(minima, fbck_position[minima, 9], "x")
        plt.legend()
    return minima[0:4]


def detect_initial_sequence_force_plate(Fz, frequency, plot=False):
    # Detect initial sequence (4 steps on the force plate)
    # Find peaks in force plate -> when the limb touches the ground
    ratio = 0.5/frequency  # width and distance of peaks are influenced by the frequency
    peaks, _ = signal.find_peaks(Fz, prominence=1, width=100*ratio, distance=150*ratio)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(Fz, label='Fz')
        ax.plot(peaks, Fz[peaks], "x")
        ax.legend()
    return peaks[0:4]


def compute_delay_between_poly_and_force_plate(fbck_position, Fxyz, frequency):
    # Detect initial sequence (4 steps)
    minima = detect_initial_sequence_polymander(fbck_position, frequency)
    peaks = detect_initial_sequence_force_plate(Fxyz[:, 2], frequency)
    # Compute offset between two signals
    offset = []
    for k in range(4):
        offset.append(peaks[k] - minima[k])
    offset = np.mean(offset)
    return offset


def manage_delay_between_poly_and_fp(t_s_poly, fbck_position, fbck_current, t_s_fp, Fxyz, frequency):
    # 1) Find delay between polymander and force plate
    delay = compute_delay_between_poly_and_force_plate(fbck_position, Fxyz, frequency)

    # 2) Cut signal to ensure that both signals are aligned according to the peaks
    # remove first seconds in force plate signal
    t_s_fp_cut_start = cut_time(t_s_fp, delay, delay + len(t_s_poly))
    Fxyz_cut_start = cut_signal(Fxyz, delay, delay + len(t_s_poly))

    # remove extra seconds in polymander
    t_s_poly_cut_end = cut_time(t_s_poly, 0, len(t_s_fp_cut_start))
    fbck_position_cut_end = cut_signal(fbck_position, 0, len(t_s_fp_cut_start))
    fbck_current_cut_end = cut_signal(fbck_current, 0, len(t_s_fp_cut_start))

    # adjust time on polymander time
    t_s_aligned = t_s_poly_cut_end

    return t_s_aligned, fbck_position_cut_end, fbck_current_cut_end, Fxyz_cut_start


def remove_initial_sequence(t_s, fbck_position, fbck_current, Fxyz, frequency):
    minima = detect_minima_of_Fz(fbck_position, frequency)
    t_s_final = cut_time(t_s, minima[4], minima[-1])
    fbck_position_final = cut_signal(fbck_position, minima[4], minima[-1])
    fbck_current_final = cut_signal(fbck_current, minima[4], minima[-1])
    Fxyz_final = cut_signal(Fxyz, minima[4], minima[-1])

    return t_s_final, fbck_position_final, fbck_current_final, Fxyz_final


def plot_aligned_signals(t_s, fbck_position, Fxyz, frequency):
    minima = detect_initial_sequence_polymander(fbck_position, frequency)
    fig, ax = plt.subplots()
    ax.set_title('Alignment of signals with initial sequence')
    ax.set(xlabel='time [s]')
    ax.plot(t_s, Fxyz[:, 2], label='Fz [N]')
    ax.plot(t_s, fbck_position[:, 9], label='9FbckPosition [rad]')
    ax.vlines(t_s[minima[0:4]], 0, max(Fxyz[:, 2]), colors='lime', linestyles='dashed', label='first 4 steps')
    ax.legend(loc='upper right')


def plot_3d_metrics(metrics, metric_name):  # metrics (kfolds x 9)
    metrics = np.array(metrics)

    results = np.mean(metrics, 1)
    errors = abs(np.max(metrics, 1) - np.min(metrics, 1))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Amplitude [rad]', labelpad=10)
    ax.set_ylabel('Frequency [Hz]', labelpad=10)
    ax.set_zlabel(metric_name)

    xlabels = np.array(['0.2', '0.35', '0.5'])
    ylabels = np.array(['0.1', '0.5', '1.0'])

    x, y = np.random.rand(2, 100) * 3
    hist, xedges, yedges = np.histogram2d(x, y, bins=3, range=[[0, 3], [0, 3]])

    # Construct arrays for the anchor positions of the 9 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")

    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 9 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = results

    # Set ticks
    ax.w_xaxis.set_ticks(ypos[0:3] + dx / 2.)
    ax.w_xaxis.set_ticklabels(xlabels)

    ax.w_yaxis.set_ticks(ypos[0:3] + dy / 2.)
    ax.w_yaxis.set_ticklabels(ylabels)

    # Set colors
    values = np.linspace(0.2, 1., xpos.ravel().shape[0])
    colors = cm.rainbow(values)

    # Plot 3D bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.6, color=colors)

    # Plot error bars
    x_error = [[x + dx] * 3 for x in range(3)]
    x_error = [item for sublist in x_error for item in sublist]
    y_error = [(y + dy) % 3 for y in range(9)]

    for (i, j, k, e) in zip(x_error, y_error, dz, errors):
        ax.errorbar(i, j, k, e, color='black', capsize=4)

    ax.view_init(30, 130)


