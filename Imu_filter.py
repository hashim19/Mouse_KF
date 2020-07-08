#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math


#imu filter function
def imu_filter(x, filter='butter', filter_order=2, cut_off_freq=10, filter_type='low', sampling_freq=20):

    """
    x: input array to be filtered
    
    filter: use 'butter' for butterworth filter. Only implemented butterworth filter for now. It is possible to implement more.
    
    filter_order: order of the butterworth flter. 

    cut_off_freq: cut off frequency for filtering. If filter_type is low (lowpass), the filter function filters all the frequencies greater than 
                    cut_off_freq and if filter_type is hp (highpass), the function filters all frequencies below the cut_off_freq. cut_off_freq 
                    must be less than half the sampling_freq.

    sampling_freq: sampling freq of the imu.
    """

    Wn = cut_off_freq / (0.5*sampling_freq)
    if filter=='butter':
        sos = signal.butter(filter_order, Wn, filter_type, fs=sampling_freq, output='sos')
        
        y = signal.sosfiltfilt(sos, x)

        return y


def imu_LowPass(xt, yt_minus_1, alpha):

    """
    xt: input data array to be filtered

    yt_minus_1: previous filtered signal

    alpha: smoothing factor

    """
    if yt_minus_1 is not None:
        yt = alpha*xt + (1-alpha)*yt_minus_1
    else:
        return xt

    return yt

