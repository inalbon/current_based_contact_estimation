import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from load_data import LoadData
from feature_engineering import *


poly_walking_FL = LoadData()
poly_walking_FL.load_polymander_data(dir_name=f'logs_polymander/walking/FL', log_name='robot_data_log_2022-11-03_10_29_33')
poly_walking_FL.load_force_plates_data(dir_name=f'logs_force_plates/walking/FL', log_name='exp1')
print(f'{len(poly_walking_FL.list_polymander)} files in list_polymander')
print(f'{len(poly_walking_FL.list_force_plates)} files in list_force_plate')

check = False
for (i, j) in zip(poly_walking_FL.list_polymander, poly_walking_FL.list_force_plates):
    i.plot_fbck_position(i.t_s, i.fbck_position_data)

    # Signal processing
    fbck_current_filtered = filtering_signal(i.fbck_current_data, 10)
    Fxyz_filtered = filtering_signal(j.Fxyz, 20)
    i.plot_fbck_current(i.t_s, fbck_current_filtered)
    j.plot_forces(j.t_s, Fxyz_filtered)

    #t_s_poly_resampled, fbck_current_resampled = resample_signal(fbck_current_filtered, i.t_s, j.t_s)
    t_s_fp_resampled, Fxyz_resampled = resample_signal(Fxyz_filtered, j.t_s, i.t_s)

    #i.plot_fbck_current(t_s_poly_resampled, fbck_current_resampled)
    j.plot_forces(t_s_fp_resampled, Fxyz_resampled)

    fbck_current_final = fbck_current_filtered
    Fxyz_final = Fxyz_resampled

    # Store data in X and y
    if check is False:
        X_final_test = fbck_current_final[:, 8:10]
        y_final_test = Fxyz_final[:, 2]
        check = True
    elif check is True:
        X_final_test = np.concatenate((X_final_test, fbck_current_final[:, 8:10]))
        y_final_test = np.concatenate((y_final_test, Fxyz_final[:, 2]))

    plt.show()