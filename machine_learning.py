import matplotlib.pyplot as plt
import numpy as np
from load_data import LoadData
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
    # print(f'Begin analysis of {i.log_name} and {j.log_name}')

    # 1) Resampling force plate signal based on polymander signal
    j.resample(i.t_s)
    # 2) Filter signal
    # i.filtering_signal()
    j.filtering_signal()

    # 3) Manage delay
    # 3.1) Detect initial sequence (4 steps)
    # Find minima in roll motor (when the limb touch the force plate)
    minima = i.detect_initial_sequence(i.frequency)
    # Find peaks in force plate (when the limb touch the force plate)
    peaks = j.detect_initial_sequence(i.frequency)

    offset = []
    for k in range(4):
      offset.append(peaks[k] - minima[k])
    offset = np.mean(offset)

    # 3.2) Cut signal to ensure that both signals are aligned according to the peaks
    j.cut_signal(offset, offset + len(i.t_s))  # remove first seconds in force plate signal
    i.cut_signal(0, len(j.t_s))  # remove extra seconds in polymander
    j.t_s = i.t_s  # adjust force plate time on polymander time

    minima = i.detect_initial_sequence(i.frequency)
    fig, ax = plt.subplots()
    ax.set_title(j.log_name)
    ax.set(xlabel='time [s]')
    ax.plot(i.t_s, j.Fxyz[:, 2], label='Fz [N]')
    ax.plot(i.t_s, i.fbck_position_data[:, 9], label='fbck position of limb [rad]')
    ax.vlines(i.t_s[minima[0:4]], 0, max(j.Fxyz[:, 2]), colors='lime', linestyles='dashed', label='first 4 steps')
    ax.legend(loc='upper right')
    plt.savefig('figures/initial_sequence.eps', format='eps')

    # 3.3) Remove initial sequence
    i.cut_signal(minima[4], minima[-1])
    j.cut_signal(minima[4], minima[-1])
    # print(f'End analysis of {i.log_name} and {j.log_name}')

    if check is False:
        X = i.fbck_current_data
        y = j.Fxyz[:, 2]
        check = True
    elif check is True:
        X = np.concatenate((X, i.fbck_current_data))
        y = np.concatenate((y, j.Fxyz[:, 2]))

print(np.shape(X))
print(np.shape(y))
plt.figure()
plt.plot(X[:, 8:10])
plt.figure()
plt.plot(y)
# Machine learning - multiple regression
# Train and test ratio 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)
mlr = LinearRegression()
model = mlr.fit(X_train, y_train)
print(model.score(X_train, y_train))

y_pred = mlr.predict(X_test)

plt.figure()
plt.title('Fz prediction')
plt.plot(y_test[0:500], label='true value')
plt.plot(y_pred[0:500], label='pred')
plt.legend()

plt.show()



