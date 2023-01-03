import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import numpy as np


def filtering_signal(signal, sigma=6):
    filtered_signal = gaussian_filter1d(signal, sigma=sigma, axis=0)
    return filtered_signal


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


def plot_aligned_signals(t_s, fbck_position, Fxyz, frequency):
    minima = detect_initial_sequence_polymander(fbck_position, frequency)
    fig, ax = plt.subplots()
    ax.set_title('Alignment of signals with initial sequence')
    ax.set(xlabel='time [s]')
    ax.plot(t_s, Fxyz[:, 2], label='Fz [N]')
    ax.plot(t_s, fbck_position[:, 9], label='9FbckPosition [rad]')
    ax.vlines(t_s[minima[0:4]], 0, max(Fxyz[:, 2]), colors='lime', linestyles='dashed', label='first 4 steps')
    ax.legend(loc='upper right')


def remove_inital_sequence(t_s, fbck_position, fbck_current, Fxyz, frequency):
    minima = detect_minima_of_Fz(fbck_position, frequency)
    t_s_final = cut_time(t_s, minima[4], minima[-1])
    fbck_current_final = cut_signal(fbck_current, minima[4], minima[-1])
    Fxyz_final = cut_signal(Fxyz, minima[4], minima[-1])

    return t_s_final, fbck_current_final, Fxyz_final
