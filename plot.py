import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math
import os

files_clean_participant = os.listdir(os.getcwd() + "\\cleaned")

plt.figure(figsize=(15, 10))
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

arrayA = []
arrayA2 = []
arrayA3 = []

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


for participant in files_clean_participant:
    print(participant)
    files_clean_task = os.listdir(os.getcwd() + "\\cleaned\\" + participant)
    for task in files_clean_task: 
        print(task)
        file_clean_id = os.listdir(os.getcwd() + "\\cleaned\\" + participant + "\\" + task)
        for ID in file_clean_id:
            print(ID)
            file_clean_target = os.listdir(os.getcwd() + "\\cleaned\\" + participant + "\\" + task + "\\" + ID)
            i = 0
            j = 0
            for activeTarget in file_clean_target:
                print(activeTarget)
                
                if activeTarget[0:1] != ".":
                    datapath = os.getcwd() + "\\cleaned\\" + participant + "\\" + task + "\\" + ID + "\\" + activeTarget
                    figurepath = os.getcwd() + "\\cleaned\\" + participant + "\\" + task + "\\" + ID
                    data = pd.read_csv(datapath)
                    
                    
                    trial_iter = iter(range(0,int(max(data['targettrial']))))
                    print(data['AccX3'].max())
                   
                    
                    for trial in trial_iter:
                        Xtrial = data[data['targettrial']==trial];
                        if Xtrial['targettrial'].empty:
                            continue;
                            
                     
                            
                        #Position/Time Plot, all ID's and targets, for this user alone     
                        plt.figure(1)
                        plt.plot(Xtrial['time']-Xtrial['time'].iloc[0],Xtrial['PosXKalman'], alpha=0.7)
                        
                        #Hook Plot for this user alone, all IDs and targets
                        plt.figure(2)
                        #plt.plot(Xtrial['cursorx'], Xtrial['AccX3'], alpha=0.7)
                        plt.plot(Xtrial['PosXKalman'], Xtrial['AccX3'], alpha=0.7)  #This should look very similar to the one above right?
                        
                       

                    plt.figure(1)
                    plt.ylabel('time')
                    plt.xlabel('Position (m)')
                    plt.title("Position")
                    plt.savefig(figurepath + '\\position {}.png'.format(int(i)))
                    plt.close(1)
                    
                    plt.figure(2)
                    plt.ylabel('Acceleration (m/2^2)')
                    plt.xlabel('Position x (m)')
                    plt.title("Hooke X")
                    plt.savefig(figurepath + '\\Hooke-x {}.png'.format(int(i)))
                    plt.close(2)
                   
                    i = i + 1
                    
