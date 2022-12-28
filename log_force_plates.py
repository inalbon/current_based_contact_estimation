import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class LogForcePlates():
    def __init__(self, dir_path, log_name):
        self.log_name = log_name
        self.dir_path = dir_path
        self.t_s = None
        self.Fxyz = None
        self.headers = None
        self.parse_log()

    def parse_log(self):
        with open(f'{self.dir_path}/{self.log_name}') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t')
            data_with_header = list(csvreader)

        self.headers = data_with_header[17]
        data = data_with_header[19:]
        num_samples = len(data)

        # load time [s] and Forces [N] in x, y and z direction
        self.t_s = np.zeros(num_samples)
        self.Fxyz = np.zeros((num_samples, 3))
        for i in range(num_samples):
            self.t_s[i] = data[i][0]
            self.Fxyz[i, :] = data[i][1:4]

    def cut_signal(self, delay, recording_time):
        start_recording = delay
        end_recording = delay + recording_time
        self.Fxyz = self.Fxyz[int(start_recording):int(end_recording), :]
        self.t_s = self.t_s[int(delay):int(end_recording)]

    def filtering_signal(self, sigma=6):
        for i in range(3):
            self.Fxyz[:, i] = gaussian_filter1d(self.Fxyz[:, i], sigma=sigma)

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

    def plot_log(self):
        plt.figure()
        for i in range(3):
            plt.plot(self.t_s, self.Fxyz[:, i], label=self.headers[i+1])

        plt.legend()
        plt.xlabel('time [s]')
        plt.ylabel('Force [N]')
        plt.title(self.log_name)



