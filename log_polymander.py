import numpy as np
import csv
import matplotlib.pyplot as plt
import random


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
        self.parse_log()

    def parse_log(self):
        with open(f'{self.dir_path}/{self.log_name}') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            data_with_header = list(csvreader)

        # load parameters of polymander
        headers_params = data_with_header[0]
        data_params = data_with_header[1]
        print('-----------Parameters of polymander-------------')
        i = 0
        for _ in headers_params:
            print(_, '=', data_params[i])
            i += 1
        print('------------------------------------------------')

        headers = data_with_header[2]
        data = data_with_header[3:]
        num_samples = len(data)

        # load time [s], goal position [rad], feedback position [rad], feedback current [mA],
        # feedback voltage [V] and goal torque body [mA] of polymander motors

        # load headers
        self.goal_position_headers = headers[2:2+self.num_motors]
        self.fbck_position_headers = headers[2+self.num_motors:2+2*self.num_motors]
        self.fbck_current_headers = headers[2+2*self.num_motors:2+3*self.num_motors]
        self.fbck_voltage_headers = headers[2+3*self.num_motors:2+4*self.num_motors]
        self.goal_torque_body_headers = headers[2+4*self.num_motors:2+4*self.num_motors+self.num_body_motors]

        # load data
        self.t_s = np.zeros(num_samples)
        self.goal_position_data = np.zeros((num_samples, self.num_motors))
        self.fbck_position_data = np.zeros((num_samples, self.num_motors))
        self.fbck_current_data = np.zeros((num_samples, self.num_motors))
        self.fbck_voltage_data = np.zeros((num_samples, self.num_motors))
        self.goal_torque_body_data = np.zeros((num_samples, self.num_body_motors))
        for i in range(num_samples):
            self.t_s[i] = float(data[i][0]) - float(data[0][0]) + (float(data[i][1]) - float(data[0][1]))*1e-6
            self.goal_position_data[i, :] = data[i][2:2+self.num_motors]
            self.fbck_position_data[i, :] = data[i][2+self.num_motors:2+2*self.num_motors]
            self.fbck_current_data[i, :] = data[i][2+2*self.num_motors:2+3*self.num_motors]
            self.fbck_voltage_data[i, :] = data[i][2+3*self.num_motors:2+4*self.num_motors]
            self.goal_torque_body_data[i, :] = data[i][2+4*self.num_motors:2+4*self.num_motors+self.num_body_motors]

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





