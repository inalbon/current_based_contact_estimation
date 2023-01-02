import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
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

    def cut_signal(self, start, end):
        start = int(start)
        end = int(end)
        self.Fxyz = self.Fxyz[start:end, :]
        self.t_s = self.t_s[start:end]

    def filtering_signal(self, sigma=6):
        self.Fxyz = gaussian_filter1d(self.Fxyz, sigma=sigma, axis=0)

    def resample(self, t_poly):
        #print(f'Recording time of robot: [{t_poly[0]}, {t_poly[-1]}]')
        #print(f'Recording time of force plate: [{self.t_s[0]}, {self.t_s[-1]}]')

        # Find number of samples in 60 s in robot data
        index = 0
        for _ in t_poly:
            if t_poly[index] >= self.t_s[-1]:
                nb_steps = index + 1
                break
            index += 1
        #print(f'Recording time of robot when cut: [{self.t_s[0]}, {t_poly[nb_steps - 1]}]')

        self.Fxyz, self.t_s = signal.resample(self.Fxyz, nb_steps, self.t_s)

    def detect_initial_sequence(self, plot=False):
        # Detect initial sequence (4 steps on the force plate)
        # Find peaks in force plate
        peaks, _ = signal.find_peaks(self.Fxyz[:, 2], prominence=1, width=100)  # or height=4, distance=200
        if plot:
            plt.figure()
            plt.plot(self.Fxyz[:, 2], label=self.headers[2])
            plt.plot(peaks, self.Fxyz[peaks, 2], "x")
        # print('Fz peaks =', peaks[0:4])
        return peaks[0:4]

    def plot_log(self):
        plt.figure()
        for i in range(3):
            plt.plot(self.t_s, self.Fxyz[:, i], label=self.headers[i+1])

        plt.legend()
        plt.xlabel('time [s]')
        plt.ylabel('Force [N]')
        plt.title(self.log_name)



