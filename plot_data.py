import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from Imu_filter import imu_filter

# define rotations 
Rz = R.from_euler('z', -180, degrees=True).as_matrix()
Rx = R.from_euler('x', -90, degrees=True).as_matrix()
R_pos = np.matmul(Rx, Rz)

# read data 
filename = 'x-direction'

data = pd.read_csv(filename + '.csv', sep=";")

print(data)

# adjust units and direction of position
data['cursorx'] = data['cursorx'] * 0.000265
data['cursory'] = data['cursory'] * 0.000265

# plot accelerations

# filter acceleration
accel_x = (data['AccX'].to_numpy()/16384) * 9.81
accel_y = (data['AccY'].to_numpy()/16384) * 9.81

######## butterworth ########
accel_x_filtered = imu_filter(accel_x, cut_off_freq=3, filter_type='low')
accel_y_filtered = imu_filter(accel_y, cut_off_freq=3, filter_type='low')

# save back to imu data frame
data['AccX'] = accel_x_filtered
data['AccY'] = accel_y_filtered

plt.figure(figsize=(12,9))

plt.plot(accel_x_filtered, label='Acceleration Along X')
plt.plot(accel_y_filtered, label='Acceleration Along Y')

plt.title(filename, fontsize = 18, fontweight = 'bold' )
plt.xlabel('Datapoints', fontsize = 16)
plt.ylabel('Acceleration Magnitude', fontsize = 16)

plt.grid()
plt.legend()

# plot position
plt.figure(figsize=(12,9))

pos_x = data['cursorx'].to_numpy().T
pos_y = data['cursory'].to_numpy().T

plt.plot(pos_x, label='Position X')
plt.plot(pos_y, label='Postion Y')

plt.title(filename, fontsize = 18, fontweight = 'bold' )
plt.xlabel('Datapoints', fontsize = 16)
plt.ylabel('Position', fontsize = 16)

plt.grid()
plt.legend()

plt.show()