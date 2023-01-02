import os
from os.path import dirname, realpath, join
from log_force_plates import LogForcePlates
from log_polymander import LogPolymander


class LoadData():
    # load files of a folder and store them in a list
    def __init__(self):
        self.list_polymander = []
        self.list_force_plates = []

    def load_polymander_data(self, dir_name, log_name=None):
        dir_path = join(f"{dirname(realpath(__file__))}/{dir_name}")
        if log_name is None:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".csv"):
                        self.list_polymander.append(LogPolymander(root, file))

        else:
            log_name = f'{log_name}.csv'
            self.list_polymander.append(LogPolymander(dir_path, log_name))

    def load_force_plates_data(self, dir_name, log_name=None):
        dir_path = join(f"{dirname(realpath(__file__))}/{dir_name}")
        if log_name is None:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".txt"):
                        self.list_force_plates.append(LogForcePlates(root, file))

        else:
            log_name = f'{log_name}.txt'
            self.list_force_plates.append(LogForcePlates(dir_path, log_name))

    def load_data(self, data_type=None, log_name=None):
        dir_path = join(f"{dirname(realpath(__file__))}/{self.dir_name}")
        if log_name is None:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if data_type == 'poly':
                        if file.endswith(".csv"):
                            self.list_polymander.append(LogPolymander(root, file))
                    elif data_type == 'force_plate':
                        if file.endswith(".txt"):
                            self.list_force_plates.append(LogForcePlates(root, file))

        else:
            if data_type == 'poly':
                self.list_polymander.append(LogPolymander(dir_path, f'{log_name}.csv'))
            elif data_type == 'force_plate':
                self.list_force_plates.append(LogForcePlates(dir_path, f'{log_name}.txt'))

