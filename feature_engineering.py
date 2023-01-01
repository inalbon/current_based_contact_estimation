import matplotlib.pyplot as plt
from load_data import LoadData
from sklearn import linear_model
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from scipy import stats
import numpy as np


class FeatureEngineering():
    def filtering_signal(self, signal, sigma=6):
        filtered_signal = gaussian_filter1d(signal, sigma=sigma, axis=0)
        return filtered_signal

