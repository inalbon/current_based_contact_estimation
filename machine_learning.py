import matplotlib.pyplot as plt
import numpy as np
from load_data import LoadData
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
list_X = LoadData(dir_name=all)
list_y = LoadData(dir_name=all)

features = 'currents'
target = 'Fz'

# Features engineering
for X in list_X:
  X = np.concatenate(X)
  y = np.concatenate(y)

# Machine learning - linear regression
# Train and test ratio 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)
regr = LinearRegression()
regr.fit(X_train, y_train)
regr.predict(X_test)



