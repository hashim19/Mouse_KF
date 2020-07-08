# -*- coding: utf-8 -*-
"""
run the script to do Kalman Filtering for mouse tracking.

Configuration Parameters:
    accel_var:      Acceleromter variance. It is a numpy array of two elements. Provide accelerometr variance for x and y axis.
    pos_var:        position measurement variance. It also a numpy array of two elements for each axis.
                    If pos_var is small, the kalman filter will have more belief on the position measurements. 

    process_var:    process var. If this value is small, the kalman filter will have more believe on the system model than measuremnts.

    filename_data:  provide path to the data file.

Kalman Filter has a state variable, which can be accessed by KalmanFilter.x. it has [pos_x, pos_y, vel_x, vel_y, accel_x, accel_y] as its elements.
Access the required variable accordingly. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymap3d
import math
import time
from scipy.spatial.transform import Rotation as R

from KF import KalmanFilter
from Imu_filter import imu_filter
from plot_func import plot_pos
import scipy.signal
import os



files_raw = os.listdir(os.getcwd() + "\\raw")
directory_cleaned = os.getcwd() + "\\cleaned"

if not os.path.exists(directory_cleaned):
    os.makedirs(directory_cleaned)
print("Verarbeitung gestartet")
for file in files_raw:
    
    print("Aktuelle Datei: " + file)
    data = pd.read_csv(os.getcwd() + "\\raw\\" + file)
    
    
    ti = pd.to_datetime(data.time*1000000000)
    data = data.set_index(ti)
    data.index = data.index.rename('t')

    data = data.resample('1ms').ffill()
    data.time = data.index.minute*60+data.index.second+data.index.microsecond/1000000.0
    data = data[1:]
    data['cursorx'] = data['cursorx']*0.000265
    data['AccX'] = data['AccX']/16384*9.8
    data['AccY'] = data['AccY']/16384*9.8
    
    taskid = data['ID'].values[10]
    participant = data['participant'].values[10]
    participant_dir = directory_cleaned + "\\P" + str(int(participant))
    if not os.path.exists(participant_dir):
        os.makedirs(participant_dir)
        os.makedirs(participant_dir + "\\ISO")
        os.makedirs(participant_dir + "\\DISCRETE")
        
    data = data[data.logData != 0]

    for i, gg in data.groupby('activeTarget'):
        gt = gg.copy()
        gt['meantrialtime'] = gt.loc[(gt['button'] == 1) & (gt['trialtime'] != -1), 'trialtime'].sum() / gt.loc[(gt['button'] == 1) & (gt['trialtime'] != -1), 'trialtime'].count()
        gt['errorrate'] = gt.loc[(gt['trialsuccess'] == 0), 'trialsuccess'].count() / gt.loc[(gt['trialsuccess'] != -1), 'trialsuccess'].count()
        for ii in gt.targettrial.unique():
            
            
            

                        #################### confgiuration parameters #####################

            # accelerometer variance
            accel_var = np.array([1, 1])

            # position measurement variance
            pos_var = np.array([0.001, 0.001])

            # proccess noise
            process_var = 0.1

            
        

            ####################### Kalman Filter Initialization #####################
            # Initialization

            kf = KalmanFilter(dim_x=6, dim_z=4)
            dt = 0.005
            kf.x = np.array([0, 0, 0, 0, 0, 0]) # location and velocity
            kf.A = np.array([[1.0, 0.0, dt, 0.0, (dt**2)/2, 0.0],
                            [0.0, 1.0, 0.0, dt, 0.0, (dt**2)/2],
                            [0.0, 0.0, 1.0, 0.0, dt, 0],
                            [0.0, 0.0, 0.0, 1.0, 0, dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])  # state transition matrix

            kf.H = np.array([[1, 0.0, 0.0, 0.0, 0, 0],
                            [0.0, 1, 0.0, 0.0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])    # Measurement function

            kf.R = np.diag(np.hstack((pos_var, accel_var)))  
                               # measurement uncertainty

            kf.P[:] = np.diag([5, 5, 10, 10, 5, 5])               # [:] makes deep copy

            sigma = process_var
            kf.Q[:] = np.array([[0, 0.0, 0.0, 0.0, 0, 0],
                          [0.0, 0, 0.0, 0.0, 0, 0],
                          [0.0, 0, 0.0, 0.0, 0, 0],
                          [0.0, 0, 0.0, 0.0, 0, 0],
                          [0.0, 0.0, 0, 0.0, sigma, 0],
                          [0.0, 0.0, 0.0, 0, 0, sigma]])

            # kf.B = np.array([
            #     [(dt**2)/2, 0],
            #     [0, (dt**2)/2],
            #     [dt, 0],
            #     [0, dt]
            # ])


            ################# reading data #################

            # read data

            kf_data = gt.loc[gt.targettrial==ii].copy()

            pos_track =  []

            pos_kf, cov_kf = [], []


            # accel list
            kf_accel_ls = []

            ################ filtering acceleration ##############

            # adjust units and direction of position
        

            # filter acceleration
            accel_x = (kf_data['AccX'].to_numpy()) 
            accel_y = (kf_data['AccY'].to_numpy()) 

            ######## butterworth ########
            accel_x_filtered = imu_filter(accel_x, cut_off_freq=7, filter_type='low')
            accel_y_filtered = imu_filter(accel_y, cut_off_freq=7, filter_type='low')

            # save back to imu data frame
            kf_data['AccX'] = accel_x_filtered
            kf_data['AccY'] = accel_y_filtered

            # get positions for comparison
            pos_true = kf_data[['cursorx', 'cursory']].to_numpy().T
            # pos_true[1,:] = - pos_true[1,:]

            print(kf_data)

            ################## iterate through data and call predict and update functions ################
            for idx, row in kf_data.iterrows():


                # get pos measurement
                pos = row[:2].to_numpy()
                # pos[1] = - pos[1]

                # get acceleration measurement
                accel = row[3:5].to_numpy()

                y = np.hstack((pos, accel))

                # print(y)

                kf.predict()
                kf.update(y)

                # print(kf.x)
                pos_kf.append(kf.x[:2])
                cov_kf.append(kf.P)

                kf_accel_ls.append(kf.x[4:])

            pos_kf = np.array(pos_kf).T
            cov_kf = np.array(cov_kf).T
            pos_track = pos_true

            kf_accel = np.array(kf_accel_ls).T
            print(pos_kf[:, :1])
            
            gt.loc[gt.targettrial==ii,'PosXKalman'] = pos_kf[0];
            gt.loc[gt.targettrial==ii,'PosYKalman'] = pos_kf[1];
            
            gt.loc[gt.targettrial==ii,'AccX'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii, 'AccX'], 301, 4, mode="nearest");
            gt.loc[gt.targettrial==ii,'AccY'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii, 'AccY'], 301, 4, mode="nearest");
            gt.loc[gt.targettrial==ii,'cursorx'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii, 'cursorx'], 301, 4, mode="nearest");
            gt.loc[gt.targettrial==ii,'cursory'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii, 'cursory'], 301, 4, mode="nearest");
            gt.loc[gt.targettrial==ii,'AccX2'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii, 'cursorx'], 301, 4, deriv=1, mode="nearest", delta=0.001);
            gt.loc[gt.targettrial==ii,'AccY2'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii,'cursory'], 301, 4, deriv=1, mode="nearest", delta=0.001);
            gt.loc[gt.targettrial==ii,'AccX3'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii,'cursorx'], 301, 4, deriv=2, mode="nearest", delta=0.001);
            gt.loc[gt.targettrial==ii,'AccY3'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii,'cursory'], 301, 4, deriv=2, mode="nearest", delta=0.001);
            gt.loc[gt.targettrial==ii,'AccXCombined'] = ((gt.loc[gt.targettrial==ii, 'AccX']*18)+gt.loc[gt.targettrial==ii, 'AccX3'])/2
        if file[0:1] == "D":
            if not os.path.exists(participant_dir + '\\DISCRETE\\ID' + str(int(taskid))):
                os.makedirs(participant_dir + '\\DISCRETE\\ID' + str(int(taskid)))
            gt.to_csv(participant_dir + '\\DISCRETE\\ID' + str(int(taskid)) + '\\{}.csv'.format(int(i)))
        if file[0:1] == "R":
            if not os.path.exists(participant_dir + '\\ISO\\ID' + str(int(taskid))):
                os.makedirs(participant_dir + '\\ISO\\ID' + str(int(taskid)))
            gt.to_csv(participant_dir + '\\ISO\\ID' + str(int(taskid)) + '\\{}.csv'.format(int(i)))