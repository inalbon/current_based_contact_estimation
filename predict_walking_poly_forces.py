import numpy as np
from load_data import LoadData
from feature_engineering import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (train_test_split, KFold, cross_val_score)
from sklearn.metrics import (mean_squared_error, mean_absolute_error)

import matplotlib.pyplot as plt

list_folder = ['amp_0.2_freq_0.1', 'amp_0.2_freq_0.5', 'amp_0.2_freq_1.0',
               'amp_0.35_freq_0.1', 'amp_0.35_freq_0.5', 'amp_0.35_freq_1.0',
               'amp_0.5_freq_0.1', 'amp_0.5_freq_0.5', 'amp_0.5_freq_1.0']

list_metrics_name = ['R2', 'MSE', 'RMSE', 'MAE']

# -------------------------------- Load data ----------------------------------------
# One limb for training model
FL_limb = LoadData()
for folder in list_folder:
    FL_limb.load_polymander_data(dir_name=f'logs_polymander/one_limb/FL/{folder}')
    FL_limb.load_force_plates_data(dir_name=f'logs_force_plates/one_limb/FL/{folder}')
print(f'{len(FL_limb.list_polymander)} files in list_polymander')
print(f'{len(FL_limb.list_force_plates)} files in list_force_plate')

# Walking polymander for prediction
poly_walking_FL = LoadData()
poly_walking_FL.load_polymander_data(dir_name=f'logs_polymander/walking/FL', log_name='robot_data_log_2022-11-03_10_29_33')
poly_walking_FL.load_force_plates_data(dir_name=f'logs_force_plates/walking/FL', log_name='exp1')
print(f'{len(poly_walking_FL.list_polymander)} files in list_polymander')
print(f'{len(poly_walking_FL.list_force_plates)} files in list_force_plate')


# -------------------------------- Supervised learning ----------------------------
check = False
for (i, j) in zip(FL_limb.list_polymander, FL_limb.list_force_plates):
    # Signal processing
    t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = signal_processing(i, j)

    # Store data in X and y
    if check is False:
        X = fbck_current_final[:, 8:10]
        y = Fxyz_final[:, 2]
        #i.plot_fbck_current(t_s_final, fbck_current_final)
        #j.plot_forces(t_s_final, Fxyz_final)
        check = True
    elif check is True:
        X = np.concatenate((X, fbck_current_final[:, 8:10]))
        y = np.concatenate((y, Fxyz_final[:, 2]))

# Train and test ratio 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

# Multiple linear regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred = mlr.predict(X_test)

# Metrics
r2 = mlr.score(X_train, y_train)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print('\n----------------------------- metrics -------------------------------')
print(f'mlr metrics : [R^2, MSE, RMSE, MAE] = [{r2:.2f}, {mse:.2f}, {rmse:.2f}, {mae:.2f}]')

# -------------------------- Prediction of Fz for FL limb of walking polymander ---------------
# Feature engineering
check = False
for (i, j) in zip(poly_walking_FL.list_polymander, poly_walking_FL.list_force_plates):
    # Plot fbck positions
    i.plot_fbck_position(i.t_s, i.fbck_position_data)

    # Filtering
    fbck_current_filtered = filtering_signal(i.fbck_current_data, 10)
    Fxyz_filtered = filtering_signal(j.Fxyz, 20)

    # Resampling polymander from 166 Hz to 100 Hz
    # (careful the t_s_poly_resampled is not accurate since period of polymander is not reliable)
    t_s_poly_resampled, fbck_current_resampled = resample_signal(fbck_current_filtered, i.t_s, j.t_s)

    # Cut force plate signal
    t_s_final = cut_time(j.t_s, 0, len(t_s_poly_resampled))
    Fxyz_final = cut_signal(Fxyz_filtered, 0, len(t_s_poly_resampled))
    fbck_current_final = fbck_current_resampled

    # Plot final fbck current and forces
    i.plot_fbck_current(t_s_final, fbck_current_final)
    j.plot_forces(t_s_final, Fxyz_final)

    plot_3d_curents_time(t_s_final, fbck_current_final[:, 8], fbck_current_final[:, 9])
    plot_3d_currents_force(Fxyz_final[:, 2], fbck_current_final[:, 8], fbck_current_final[:, 9])

    # Store data in X and y
    if check is False:
        X_final_test = fbck_current_final[:, 8:10]
        y_final_test = Fxyz_final[:, 2]
        check = True
    elif check is True:
        X_final_test = np.concatenate((X_final_test, fbck_current_final[:, 8:10]))
        y_final_test = np.concatenate((y_final_test, Fxyz_final[:, 2]))

X_test_walking = X_final_test
y_test_walking = y_final_test
y_pred_walking = mlr.predict(X_test_walking)

plot_3d_currents_force(y_pred_walking, fbck_current_final[:, 8], fbck_current_final[:, 9])


# Plot prediction of Fz when polymander is walking
fig, ax = plt.subplots()
ax.set_title('Prediction of Fz with hip and calf motors')
ax.plot(y_test_walking, label='true value')
ax.plot(y_pred_walking, label='pred')
ax.set(xlabel='time [s]', ylabel='Fz [N]')
ax.legend()

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
print("R2: mean score of %0.4f with a standard deviation of %0.4f" % (r2_scores.mean(), r2_scores.std()))
print("MSE: mean score of %0.4f with a standard deviation of %0.4f" % (mse_scores.mean(), mse_scores.std()))
print("RMSE: mean score of %0.4f with a standard deviation of %0.4f" % (rmse_scores.mean(), rmse_scores.std()))
print("MAE: mean score of %0.4f with a standard deviation of %0.4f\n" % (mae_scores.mean(), mae_scores.std()))

plt.show()







