import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pandas as pd


class LogPolymander():
    def __init__(self, dir_path, log_name):
        self.log_name = log_name
        self.dir_path = dir_path
        self.t_s = None
        self.goal_position_data = None
        self.goal_position_headers = None
        self.fbck_position_data = None
        self.fbck_position_headers = None
        self.fbck_current_data = None
        self.fbck_current_headers = None
        self.fbck_voltage_data = None
        self.fbck_voltage_headers = None
        self.goal_torque_body_data = None
        self.goal_torque_body_headers = None
        self.num_motors = 16
        self.num_body_motors = 8
        self.frequency = None

        self.parse_log()

    def parse_log(self):
        # load parameters of polymander
        params = pd.read_csv(f'{self.dir_path}/{self.log_name}', nrows=1)
        print('-----------Parameters of polymander-------------')
        print(params)
        self.frequency = params.values[0, 3]

        # load time [s], goal position [rad], feedback position [rad], feedback current [mA],
        # feedback voltage [V] and goal torque body [mA] of polymander motors
        df = pd.read_csv(f'{self.dir_path}/{self.log_name}', skiprows=2, dtype='float64')
        num_samples = df.shape[0]

        # load headers
        self.goal_position_headers = df.columns[2:2+self.num_motors]
        self.fbck_position_headers = df.columns[2+self.num_motors:2+2*self.num_motors]
        self.fbck_current_headers = df.columns[2+2*self.num_motors:2+3*self.num_motors]
        self.fbck_voltage_headers = df.columns[2+3*self.num_motors:2+4*self.num_motors]
        self.goal_torque_body_headers = df.columns[2+4*self.num_motors:2+4*self.num_motors+self.num_body_motors]

        # load data
        self.t_s = np.zeros(num_samples)
        self.goal_position_data = df.values[:, 2:2+self.num_motors]
        self.fbck_position_data = df.values[:, 2+self.num_motors:2+2*self.num_motors]
        self.fbck_current_data = df.values[:, 2+2*self.num_motors:2+3*self.num_motors]
        self.fbck_voltage_data = df.values[:, 2+3*self.num_motors:2+4*self.num_motors]
        self.goal_torque_body_data = df.values[:, 2+4*self.num_motors:2+4*self.num_motors+self.num_body_motors]
        for i in range(num_samples):
            self.t_s[i] = df.values[i][0] - df.values[0][0]\
                          + (df.values[i][1] - df.values[0][1])*1e-6

    def filtering_signal(self, sigma=6):
        self.goal_position_data = gaussian_filter1d(self.goal_position_data, sigma=sigma, axis=0)
        self.fbck_position_data = gaussian_filter1d(self.fbck_position_data, sigma=sigma, axis=0)
        self.fbck_current_data = gaussian_filter1d(self.fbck_current_data, sigma=sigma, axis=0)
        self.fbck_voltage_data = gaussian_filter1d(self.fbck_voltage_data, sigma=sigma, axis=0)
        self.goal_torque_body_data = gaussian_filter1d(self.goal_torque_body_data, sigma=sigma, axis=0)


    def cut_signal(self, start, end):
        start = int(start)
        end = int(end)
        self.t_s = self.t_s[start:end]
        self.goal_position_data = self.goal_position_data[start:end, :]
        self.fbck_position_data = self.fbck_position_data[start:end, :]
        self.fbck_current_data = self.fbck_current_data[start:end, :]
        self.fbck_voltage_data = self.fbck_voltage_data[start:end, :]
        self.goal_torque_body_data = self.goal_torque_body_data[start:end, :]

    def detect_initial_sequence(self, frequency, plot=False):
        ratio = 0.5/frequency
        minima, _ = signal.find_peaks(-self.fbck_position_data[:, 9], prominence=-0.5, width=100*ratio, distance=150*ratio)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.fbck_position_data[:, 9], label=self.fbck_position_headers[9])
            ax.plot(minima, self.fbck_position_data[minima, 9], "x")
            plt.legend()
        # print('Fbck pos. minima =', minima[0:4])
        return minima

    def plot_goal_position(self):
        self.plot_limbs(self.goal_position_data, self.goal_position_headers)
        #self.plot_spine(self.goal_position_data, self.goal_position_headers)

    def plot_fbck_position(self):
        self.plot_limbs(self.fbck_position_data, self.fbck_position_headers)
        #self.plot_spine(self.fbck_position_data, self.fbck_position_headers)

    def plot_fbck_current(self):
        self.plot_limbs(self.fbck_current_data, self.fbck_current_headers)
        #self.plot_spine(self.fbck_current_data, self.fbck_current_headers)


    def plot_fbck_voltage(self):
        self.plot_limbs(self.fbck_voltage_data, self.fbck_voltage_headers)
        #self.plot_spine(self.fbck_voltage_data, self.fbck_voltage_headers)

    def plot_goal_body_torque(self):
        self.plot_spine(self.goal_torque_body_data, self.goal_torque_body_headers)

    def plot_position_vs_current(self, i):
        fig, ax = plt.subplots()
        fig.suptitle(self.log_name)
        ax.plot(self.t_s, self.fbck_position_data[:, i], label=self.fbck_position_headers[i])
        ax.plot(self.t_s, self.fbck_current_data[:, i]*1e-3, label=self.fbck_current_headers[i])
        ax.legend()

    def plot_limbs(self, data, headers):
        # plot limbs joints (motors ID: 9-16)
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.suptitle(self.log_name)
        for i in [8, 9]:
            axs[0, 0].plot(self.t_s, data[:, i], label=headers[i])
            axs[0, 0].set_title('Front left limb')

        for i in [10, 11]:
            axs[0, 1].plot(self.t_s, data[:, i], label=headers[i])
            axs[0, 1].set_title('Front right limb')

        for i in [12, 13]:
            axs[1, 0].plot(self.t_s, data[:, i], label=headers[i])
            axs[1, 0].set_title('Hind left limb')

        for i in [14, 15]:
            axs[1, 1].plot(self.t_s, data[:, i], label=headers[i])
            axs[1, 1].set_title('Hind right limb')

        for ax in axs.flat:
            ax.set(xlabel='time [s]', ylabel=headers[0][1:])
            ax.legend()
            ax.label_outer()

    def plot_spine(self, data, headers):
        # plot body joints in 8 subplots (motors ID: 1-8)
        #fig, axs = plt.subplots(2, 4)
        #fig.suptitle(self.log_name)

        #i = 0
        #for ax in axs.flat:
        #    ax.plot(self.t_s, data[:, i], label=headers[i])
        #    ax.set(xlabel='time [s]', ylabel=headers[0][1:])
        #    ax.legend()
        #    ax.label_outer()
        #    i += 1

        # plot body joints in one plot (motors ID: 1-8)
        fig, ax = plt.subplots()
        fig.suptitle(self.log_name)
        for i in range(8):
            ax.plot(self.t_s, data[:, i], label=headers[i])
            ax.set(xlabel='time [s]', ylabel=headers[0][1:])
            ax.legend()





