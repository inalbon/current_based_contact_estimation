import numpy as np
from load_data import LoadData
from feature_engineering import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (train_test_split, KFold, cross_val_score)
from sklearn.metrics import (mean_squared_error, mean_absolute_error)

import matplotlib.pyplot as plt
from matplotlib import cm

list_folder = ['amp_0.2_freq_0.1', 'amp_0.2_freq_0.5', 'amp_0.2_freq_1.0',
               'amp_0.35_freq_0.1', 'amp_0.35_freq_0.5', 'amp_0.35_freq_1.0',
               'amp_0.5_freq_0.1', 'amp_0.5_freq_0.5', 'amp_0.5_freq_1.0']

list_r2 = []
list_mse = []
list_rmse = []
list_mae = []
list_metrics_name = ['R2', 'MSE', 'RMSE', 'MAE']

# -------------------------------- Load data ----------------------------------------
load_data = LoadData()
for folder in list_folder:
    load_data.load_polymander_data(dir_name=f'logs_polymander/one_limb/FL/{folder}')
    load_data.load_force_plates_data(dir_name=f'logs_force_plates/one_limb/FL/{folder}')
print(f'{len(load_data.list_polymander)} files in list_polymander')
print(f'{len(load_data.list_force_plates)} files in list_force_plate')

# -------------------------------- Feature engineering ----------------------------
check = False
for (i, j) in zip(load_data.list_polymander, load_data.list_force_plates):
    # Signal processing
    t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = signal_processing(i, j)

    # Store data in X and y
    if check is False:
        X = fbck_current_final[:, 8:10]
        y = Fxyz_final[:, 2]
        check = True
    elif check is True:
        X = np.concatenate((X, fbck_current_final[:, 8:10]))
        y = np.concatenate((y, Fxyz_final[:, 2]))

# ------------------------ Supervised learning ----------------------------------
# Train and test ratio 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

# Multiple linear regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred = mlr.predict(X_test)

# Metrics
# r2 = mlr.score(X_train, y_train)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# print('\n----------------------------- metrics -------------------------------')
# print(f'mlr metrics : [R^2, MSE, RMSE, MAE] = [{r2:.2f}, {mse:.2f}, {rmse:.2f}, {mae:.2f}]')

# K-fold cross validation
folds = KFold(n_splits=5, shuffle=True, random_state=100)

r2_scores = cross_val_score(mlr, X_train, y_train, scoring='r2', cv=folds)
mse_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_mean_squared_error', cv=folds)
rmse_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_root_mean_squared_error', cv=folds)
mae_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_mean_absolute_error', cv=folds)

mse_scores = abs(mse_scores)
rmse_scores = abs(rmse_scores)
mae_scores = abs(mae_scores)

print('\n------------------------ k-fold cross validation----------------------')
#print(f'R2 of {folder} = {r2_scores}')
print("R2: mean score of %0.4f with a standard deviation of %0.4f" % (r2_scores.mean(), r2_scores.std()))
#print(f'MSE of {folder} = {mse_scores}')
print("MSE: mean score of %0.4f with a standard deviation of %0.4f" % (mse_scores.mean(), mse_scores.std()))
#print(f'RMSE of {folder} = {rmse_scores}')
print("RMSE: mean score of %0.4f with a standard deviation of %0.4f" % (rmse_scores.mean(), rmse_scores.std()))
#print(f'MAE of {folder} = {mae_scores}')
print("MAE: mean score of %0.4f with a standard deviation of %0.4f\n" % (mae_scores.mean(), mae_scores.std()))
