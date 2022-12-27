import pandas as pd
import csv
from sklearn import linear_model

file = 'robot_data_log_2022-12-23_20_23_54.csv'
file2 = 'exp1_amp_0.2_freq_0.1.txt'
skiprows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]

df = pd.read_csv(file, skiprows=2)
df2 = pd.read_csv(file2, skiprows=skiprows, sep='\t')
features = ['8FbckCurrent[mA]', '9FbckCurrent[mA]']
target = 'Fz'
X = df[features].values.reshape(-1, len(features))
y = df2[target].values.reshape(-1, 1)
print(X.shape, y.shape)

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
