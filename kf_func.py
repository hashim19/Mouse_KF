import numpy as np 
import pandas as pd 
import math

def state_transition_func(x, dt):

    # apply transition to positon
    x[:2] = x[:2] + dt*x[2:4]

    # make rotation matrix
    R = np.array([
        [np.cos(x[6]), -np.sin(x[6])],
        [np.sin(x[6]), np.cos(x[6])]
    ])

    # transform accelerations
    accel_trans = np.matmul(R, x[4:6])

    # apply transition to velocity
    x[2:4] = x[2:4] + dt*accel_trans

    # update acceleration
    x[4:6] = accel_trans

    