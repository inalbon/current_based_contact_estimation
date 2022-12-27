import os
from os.path import dirname, realpath, join
import log_force_plates
from log_polymander import LogPolymander


class LoadData():
    # load files in a folder and store them in a list
    def __init__(self, dir_name):
        self.list_polymander = []
        self.list_force_plates = []
        self.dir_name = dir_name

    def load_polymander_data(self, log_name=None):
        dir_path = join(f"{dirname(realpath(__file__))}/{self.dir_name}")
        if log_name is None:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".csv"):
                        self.list_polymander.append(LogPolymander(root, file))

        else:
            log_name = f'{log_name}.csv'
            self.list_polymander.append(LogPolymander(dir_path, log_name))

    def load_force_plates_data(self, log_name=None):
        dir_path = join(f"{dirname(realpath(__file__))}/{self.dir_name}")
        if log_name is None:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".txt"):
                        self.list_force_plates.append(log_force_plates.LogForcePlates(root, file))

        else:
            log_name = f'{log_name}.txt'
            self.list_force_plates.append(log_force_plates.LogForcePlates(dir_path, log_name))

