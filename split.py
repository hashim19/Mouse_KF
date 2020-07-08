import pandas as pd
import numpy as np
import scipy.signal
import math
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
            gt.loc[gt.targettrial==ii,'AccX'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii, 'AccX'], 301, 4, mode="nearest");
            gt.loc[gt.targettrial==ii,'AccY'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii, 'AccY'], 301, 4, mode="nearest");
            #gt.loc[gt.targettrial==ii,'cursorx'] = scipy.signal.savgol_filter(gt.loc[gt.targettrial==ii, 'cursorx'], 301, 4, mode="nearest");
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