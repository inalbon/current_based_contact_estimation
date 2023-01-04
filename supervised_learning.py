import numpy as np
from load_data import LoadData
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
from feature_engineering import *
import matplotlib.pyplot as plt

# Load data
load_data = LoadData()
load_data.load_polymander_data(dir_name='logs_polymander/one_limb/FL/amp_0.2_freq_0.1')
print('Polymander data loaded')
load_data.load_force_plates_data(dir_name='logs_force_plates/one_limb/FL/amp_0.2_freq_0.1')
print('Force plate data loaded')

features = load_data.list_polymander[0].fbck_current_headers
target = load_data.list_force_plates[0].headers[3]

# Feature engineering
check = False
count = 0
for (i, j) in zip(load_data.list_polymander, load_data.list_force_plates):
    # 1) Filter signal
    fbck_current_filtered = filtering_signal(i.fbck_current_data, 10)
    Fxyz_filtered = filtering_signal(j.Fxyz, 20)

    # 2) Resampling force plate signal based on polymander signal
    t_s_fp_resampled, Fxyz_resampled = j.resample_force_plate(Fxyz_filtered, i.t_s)

    # 3) Manage delay
    t_s_cut, fbck_position_cut, fbck_current_cut, Fxyz_cut = manage_delay_between_poly_and_fp(i.t_s,
                                                                                              i.fbck_position_data,
                                                                                              fbck_current_filtered,
                                                                                              t_s_fp_resampled,
                                                                                              Fxyz_resampled,
                                                                                              i.frequency)

    # 4) Remove initial sequence
    t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = remove_inital_sequence(t_s_cut, fbck_position_cut,
                                                                                            fbck_current_cut,
                                                                                            Fxyz_cut, i.frequency)

    # 5) Store data in X and y
    if check is False:
        X = fbck_current_final[:, 8:10]
        y = Fxyz_final[:, 2]
        check = True
    elif check is True and count != 4:
        X = np.concatenate((X, fbck_current_final[:, 8:10]))
        y = np.concatenate((y, Fxyz_final[:, 2]))
    elif check is True and count == 4:
        X_valid = fbck_current_final[:, 8:10]
        y_valid = Fxyz_final[:, 2]
    count += 1



# Train and test ratio 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

# Supervised Learning - linear regression with motor 8
x_train = X_train[:, 0]
x_test = X_test[:, 0]
x_valid = X_valid[:, 0]

lr = LinearRegression()
lr.fit(x_train.reshape(-1, 1), y_train)
r2 = lr.score(x_train.reshape(-1, 1), y_train)
print('model score of lr', r2)

y_pred = lr.predict(x_valid.reshape(-1, 1))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].set_title('Linear Regression of motor 8')
axs[0].set(xlabel='Feedback current [mA]', ylabel='Fz [N]')
axs[0].scatter(x_train, y_train)
axs[0].plot(x_train, lr.coef_*x_train + lr.intercept_, 'r', label='$R^2 = %.2f$' % r2)
axs[0].legend()

axs[1].set_title('Prediction of Fz with linear regression (motor 8)')
axs[1].plot(t_s_final, y_valid, label='true value')
axs[1].plot(t_s_final, y_pred, label='pred')
axs[1].set(xlabel='time [s]', ylabel='Fz [N]')
axs[1].legend()
plt.savefig('figures/lr_results/lr_8.png', format='png')


# Supervised Learning - linear regression with motor 9
x_train = X_train[:, 1]
x_test = X_test[:, 1]
x_valid = X_valid[:, 1]
# Linear regression with motor 9 (sklearn)
lr = LinearRegression()
lr.fit(x_train.reshape(-1, 1), y_train)
r2 = lr.score(x_train.reshape(-1, 1), y_train)
print('model score of lr', r2)

y_pred = lr.predict(x_valid.reshape(-1, 1))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].set_title('Linear Regression of motor 9')
axs[0].set(xlabel='Feedback current [mA]', ylabel='Fz [N]')
axs[0].scatter(x_train, y_train)
axs[0].plot(x_train, lr.coef_*x_train + lr.intercept_, 'r', label='$R^2 = %.2f$' % r2)
axs[0].legend()

axs[1].set_title('Prediction of Fz with linear regression (motor 9)')
axs[1].plot(t_s_final, y_valid, label='true value')
axs[1].plot(t_s_final, y_pred, label='pred')
axs[1].set(xlabel='time [s]', ylabel='Fz [N]')
axs[1].legend()

plt.savefig('figures/lr_results/lr_9.png', format='png')

# Supervised Learning - multiple linear regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)
r2 = mlr.score(X_train, y_train)
print('model score of mlr: R^2 = ', r2)

print(np.shape(mlr.coef_[0]*X_train[:, 0] + mlr.coef_[1]*X_train[:, 1] + mlr.intercept_))

y_pred = mlr.predict(X_valid)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train)
ax.plot(X_train[:, 0], X_train[:, 1], mlr.coef_[0]*X_train[:, 0] + mlr.coef_[1]*X_train[:, 1] + mlr.intercept_, 'r', label='$R^2 = %.2f$' % r2)
ax.set(xlabel='8FbckCurrent [mA]', ylabel='9FbckCurrent [mA]', zlabel='Fz [N]')
ax.legend()
plt.savefig('figures/lr_results/mlr.png', format='png')

fig, ax = plt.subplots()
ax.set_title('Multiple Linear Regression prediction')
ax.plot(t_s_final, y_valid, label='true value')
ax.plot(t_s_final, y_pred, label='pred')
ax.set(xlabel='time [s]', ylabel='Fz [N]')
ax.legend()
plt.savefig('figures/lr_results/mlr_pred.png', format='png')

plt.show()



