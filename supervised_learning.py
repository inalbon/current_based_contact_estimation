import numpy as np
from load_data import LoadData
from feature_engineering import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (train_test_split, KFold)
from sklearn.metrics import (mean_squared_error, mean_absolute_error)

import matplotlib.pyplot as plt
from matplotlib import cm

list_folder = ['amp_0.2_freq_0.1', 'amp_0.2_freq_0.5', 'amp_0.2_freq_1.0',
               'amp_0.35_freq_0.1', 'amp_0.35_freq_0.5', 'amp_0.35_freq_1.0',
               'amp_0.5_freq_0.1', 'amp_0.5_freq_0.5', 'amp_0.5_freq_1.0']
array_folder = np.array(list_folder).reshape(3, 3)
list_r2 = []
list_mse = []
list_rmse = []
list_mae = []
list_metrics = ['R2', 'MSE', 'RMSE', 'MAE']

for folder in list_folder:
    # -------------------------------- Load data ----------------------------------------
    load_data = LoadData()
    load_data.load_polymander_data(dir_name=f'logs_polymander/one_limb/FL/{folder}')
    load_data.load_force_plates_data(dir_name=f'logs_force_plates/one_limb/FL/{folder}')
    print(f'{len(load_data.list_polymander)} files loaded for polymander and '
          f'{len(load_data.list_force_plates)} files loaded for force plate')

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

    # # Linear regression with hip motor (motor 8)
    # x_train = X_train[:, 0]
    # x_test = X_test[:, 0]
    #
    # lr = LinearRegression()
    # lr.fit(x_train.reshape(-1, 1), y_train)
    # y_pred = lr.predict(x_test.reshape(-1, 1))
    #
    # # Metrics
    # r2 = lr.score(x_train.reshape(-1, 1), y_train)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_test, y_pred)
    # print(f'lr metrics for hip current (current 8): [R^2, MSE, RMSE, MAE] = [{r2:.2f}, {mse:.2f}, {rmse:.2f}, {mae:.2f}]')
    #
    # # Plot results
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # axs[0].set_title('Linear regression of hip motor')
    # axs[0].set(xlabel='Feedback current [mA]', ylabel='Fz [N]')
    # axs[0].scatter(x_train, y_train)
    # axs[0].plot(x_train, lr.coef_*x_train + lr.intercept_, 'r', label='$R^2 = %.2f$' % r2)
    # axs[0].legend()
    #
    # axs[1].set_title('Prediction of Fz with hip motor model')
    # axs[1].plot(y_test, label='true value')
    # axs[1].plot(y_pred, label='pred')
    # axs[1].set(xlabel='time [s]', ylabel='Fz [N]')
    # axs[1].legend()
    #
    # plt.savefig('figures/lr_results/lr_8.png', format='png')
    # plt.savefig('figures/lr_results/lr_8.eps', format='eps')
    #
    #
    # # Linear regression with calf motor (motor 9)
    # x_train = X_train[:, 1]
    # x_test = X_test[:, 1]
    #
    # lr = LinearRegression()
    # lr.fit(x_train.reshape(-1, 1), y_train)
    # y_pred = lr.predict(x_test.reshape(-1, 1))
    #
    # # Metrics
    # r2 = lr.score(x_train.reshape(-1, 1), y_train)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_test, y_pred)
    # print(f'lr metrics for calf current (current 9): [R^2, MSE, RMSE, MAE] = [{r2:.2f}, {mse:.2f}, {rmse:.2f}, {mae:.2f}]')
    #
    # # Plot results
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # axs[0].set_title('Linear regression of calf motor')
    # axs[0].set(xlabel='Feedback current [mA]', ylabel='Fz [N]')
    # axs[0].scatter(x_train, y_train)
    # axs[0].plot(x_train, lr.coef_*x_train + lr.intercept_, 'r', label='$R^2 = %.2f$' % r2)
    # axs[0].legend()
    #
    # axs[1].set_title('Prediction of Fz with calf motor model')
    # axs[1].plot(y_test, label='true value')
    # axs[1].plot(y_pred, label='pred')
    # axs[1].set(xlabel='time [s]', ylabel='Fz [N]')
    # axs[1].legend()
    #
    # plt.savefig('figures/lr_results/lr_9.png', format='png')
    # plt.savefig('figures/lr_results/lr_9.eps', format='eps')

    # Multiple linear regression
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    y_pred = mlr.predict(X_test)

    # Metrics
    r2 = mlr.score(X_train, y_train)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'mlr metrics : [R^2, MSE, RMSE, MAE] = [{r2:.2f}, {mse:.2f}, {rmse:.2f}, {mae:.2f}]')

    # Store values for each folder in a list
    list_r2.append(r2)
    list_mse.append(mse)
    list_rmse.append(rmse)
    list_mae.append(mae)

    # # Plot results
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(X_train[:, 0], X_train[:, 1], y_train)
    # ax.plot(X_train[:, 0], X_train[:, 1], mlr.coef_[0]*X_train[:, 0] + mlr.coef_[1]*X_train[:, 1] + mlr.intercept_, 'r', label='$R^2 = %.2f$' % r2)
    # ax.set(xlabel='8FbckCurrent [mA]', ylabel='9FbckCurrent [mA]', zlabel='Fz [N]')
    # ax.legend()
    #
    # plt.savefig('figures/lr_results/mlr.png', format='png')
    # plt.savefig('figures/lr_results/mlr.eps', format='eps')
    #
    # fig, ax = plt.subplots()
    # ax.set_title('Prediction of Fz with hip and calf motors')
    # ax.plot(y_test, label='true value')
    # ax.plot(y_pred, label='pred')
    # ax.set(xlabel='time [s]', ylabel='Fz [N]')
    # ax.legend()
    #
    # plt.savefig('figures/lr_results/mlr_pred.png', format='png')
    # plt.savefig('figures/lr_results/mlr_pred.eps', format='eps')
print(list_r2)
print(list_mse)
print(list_rmse)
print(list_mae)

total_list = [list_r2, list_mse, list_rmse, list_mae]
# Plot 3D bar charts
for (metric, temp) in zip(list_metrics, total_list):
    result = np.array(temp).reshape(3, 3)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('Frequency [Hz]', labelpad=10)
    ax1.set_ylabel('Amplitude [rad]', labelpad=10)
    ax1.set_zlabel(metric)

    xlabels = np.array(['0.1', '0.5', '1.0'])
    xpos = np.arange(xlabels.shape[0])
    ylabels = np.array(['0.2', '0.35', '0.5'])
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos=result
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos

    ax1.w_xaxis.set_ticks(xpos + dx/2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy/2.)
    ax1.w_yaxis.set_ticklabels(ylabels)

    values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
    plt.savefig(f'figures/lr_results/3D_bar_chart_{metric}.png', format='png')
    plt.savefig(f'figures/lr_results/3D_bar_chart_{metric}.eps', format='eps')

plt.show()
