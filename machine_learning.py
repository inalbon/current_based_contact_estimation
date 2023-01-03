import matplotlib.pyplot as plt
import numpy as np
from load_data import LoadData
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from feature_engineering import *

# Load data
load_data = LoadData()
load_data.load_polymander_data(dir_name='logs_polymander/one_limb/FL')
load_data.load_force_plates_data(dir_name='logs_force_plates/one_limb/FL')

# Store them in a list
list_X = load_data.list_polymander
list_y = load_data.list_force_plates

features = list_X[0].fbck_current_headers
target = list_y[0].headers[3]

# Feature engineering
check = False
for (i, j) in zip(list_X, list_y):
    # 1) Resampling force plate signal based on polymander signal
    t_s_fp_resampled, Fxyz_resampled = j.resample_force_plate(j.Fxyz, i.t_s)

    # 2) Filter signal
    # i.filtering_signal()
    Fxyz_filtered = filtering_signal(Fxyz_resampled)

    # 3) Manage delay
    t_s_cut, fbck_position_cut, fbck_current_cut, Fxyz_cut = manage_delay_between_poly_and_fp(i.t_s,
                                                                                              i.fbck_position_data,
                                                                                              i.fbck_current_data,
                                                                                              t_s_fp_resampled,
                                                                                              Fxyz_filtered,
                                                                                              i.frequency)

    # 4) Remove initial sequence
    t_s_final, fbck_current_final, Fxyz_final = remove_inital_sequence(t_s_cut, fbck_position_cut, fbck_current_cut,
                                                                       Fxyz_cut, i.frequency)

    # 5) Store data in X and y
    if check is False:
        X = fbck_current_final[:, 8:10]
        y = Fxyz_final[:, 2]
        check = True
    elif check is True:
        X = np.concatenate((X, fbck_current_final[:, 8:10]))
        y = np.concatenate((y, Fxyz_final[:, 2]))

fig, ax = plt.subplots()
ax.plot(X)

fig, ax = plt.subplots()
ax.plot(y)

# Machine learning - multiple regression
# Train and test ratio 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)
mlr = LinearRegression()
model = mlr.fit(X_train, y_train)
print(model.score(X_train, y_train))

y_pred = mlr.predict(X_test)

fig, ax = plt.subplots()
ax.set_title('Fz prediction')
ax.plot(y_test[0:500], label='true value')
ax.plot(y_pred[0:500], label='pred')
ax.legend()

plt.show()



