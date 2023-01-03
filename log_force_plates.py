import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd


class LogForcePlates():
    def __init__(self, dir_path, log_name):
        self.log_name = log_name
        self.dir_path = dir_path
        self.t_s = None
        self.Fxyz = None
        self.headers = None

        self.parse_log()

    def parse_log(self):
        # load time [s] and Forces [N] in x, y and z direction
        lines_to_skip = [i for i in range(19) if i != 17]
        df = pd.read_csv(f'{self.dir_path}/{self.log_name}', delimiter='\t', skiprows=lines_to_skip, dtype='float64')

        self.headers = df.columns
        self.t_s = df.values[:, 0]
        self.Fxyz = df.values[:, 1:4]

    def resample_force_plate(self, signal_to_resample, t_poly):
        #print(f'Recording time of robot: [{t_poly[0]}, {t_poly[-1]}]')
        #print(f'Recording time of force plate: [{self.t_s[0]}, {self.t_s[-1]}]')

        # Find number of samples in 29 s in robot data and convert to total number of force plate seconds
        indices = np.where(t_poly > 29)
        nb_steps = int(indices[0][0]*self.t_s[-1]/29)

        signal_resampled, t_s_resampled = signal.resample(signal_to_resample, nb_steps, self.t_s)
        return t_s_resampled, signal_resampled

    def plot_forces(self, t_s, Fxyz):
        fig, ax = plt.subplots()
        for i in range(3):
            plt.plot(t_s, Fxyz[:, i], label=self.headers[i+1])

        ax.legend()
        ax.set(xlabel='time [s]', ylabel='Force [N]')
        ax.set_title(self.log_name)



