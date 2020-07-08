import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


fig, ax = plt.subplots()
line2, = ax.plot([], [], "g-")
line1, = ax.plot([], [], "bo")

kf_xdata, kf_ydata = [], []
gps_xdata, gps_ydata = [], []

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    return [line1,line2]


def update_state(i, imu_time, gps_time, pos_kf, gps_track):
    
    global gps_id
    # global imu_time
    if i == 0:
    #     imu_time = 0
        gps_id = 0
    # if i > 0 and i < len(pos_kf[0,:])-1:
    #     # imu time
    #     imu_time = round((imu_data['timestamp'].iloc[i] - start_time)/1000, 3)
        # print(imu_time)
    
    # print(i)
    # print("Imu time = {}".format(imu_time[i]))
    # print("Gps time = {}".format(gps_time[gps_id]))

    if imu_time[i] == gps_time[gps_id] or abs(imu_time[i] - gps_time[gps_id]) <= 0.009:
        # update gps counter
        gps_id = min(gps_id+1, len(gps_track[0,:])-1)

        gps_xdata.append(gps_track[0,gps_id])
        gps_ydata.append(gps_track[1,gps_id])

    kf_xdata.append(pos_kf[0,i])
    kf_ydata.append(pos_kf[1,i])

    line1.set_data(gps_xdata, gps_ydata)
    line2.set_data(kf_xdata, kf_ydata)
    
    return [line1,line2]

def plot_animation(imu_time, gps_time, pos_kf, gps_track, outage_start=0, outage_stop=0, video_name = 'test_animation'):

    global min_x, max_x, min_y, max_y
    # global gps_id
    # gps_id = 0
    min_x = min(pos_kf[0,:])-1
    max_x = max(pos_kf[0,:])+1
    min_y = min(pos_kf[1,:])-1
    max_y =  max(pos_kf[1,:])+1

    anim = animation.FuncAnimation(fig, update_state, init_func=init, fargs = (imu_time, gps_time, pos_kf, gps_track),
                               frames=len(pos_kf[0,:])-1, interval=20, blit=True)

    ax.plot(gps_track[0,0], gps_track[1,0], 'gP', markersize = 8, label = 'Start Gps')
    ax.plot(gps_track[0,len(gps_track[0,:])-1], gps_track[1,len(gps_track[1,:])-1], 'rP', markersize = 8, label = 'Stop Gps')

    # if cfg.settings['rfid_outage']:
    #     outage_pos_start = gps_track[:, outage_start]
    #     outage_pos_stop = gps_track[:, outage_stop]
    #     ax.annotate('outage start', xy = (outage_pos_start[0], outage_pos_start[1]))
    #     ax.annotate('outage stop', xy = (outage_pos_stop[0], outage_pos_stop[1]))

    anim.save(video_name + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


def plot_pos(pos_kf, gps_track=None, frame='ENU'):

    # get gps data intoa numpy array
    # gps_track = gps_data[['pos_enu_x', 'pos_enu_y']].to_numpy().T

    # plot kf pos and gps track
    #setup limits of x-axis and y axis equal
    if max(pos_kf[0,:]) > max(pos_kf[1,:]):
        right = max(pos_kf[0,:]) + 2
    else:
        right = max(pos_kf[1,:]) + 2
    if min(pos_kf[0,:]) < min(pos_kf[1,:]):
        left = min(pos_kf[0,:]) - 2
    else:
        left = min(pos_kf[1,:]) - 2
    
    plt.figure(figsize=(12,9))
    
    if frame == 'ENU':

        if gps_track is None:
            plt.plot(pos_kf[0,:],pos_kf[1,:], 'g', linewidth=3, label = 'KF pos')
            plt.xlim(left, right)
            plt.ylim(left, right)
        else:

            # kf start and stop point
            plt.plot(pos_kf[0,0], pos_kf[1,0], 'gp', markersize = 8, label = 'Start KF')
            plt.plot(pos_kf[0,len(pos_kf[0,:])-1], pos_kf[1,len(pos_kf[1,:])-1], 'rp', markersize = 8, label = 'Stop KF')

            # gps start and stop point
            plt.plot(gps_track[0,0], gps_track[1,0], 'gP', markersize = 8, label = 'Start Camera Pos')
            plt.plot(gps_track[0,len(gps_track[0,:])-1], gps_track[1,len(gps_track[1,:])-1], 'rP', markersize = 8, label = 'Stop Camera Pos')

            # if settings.loc["gnss_outage"].Value == 'on':
            #     outage_pos_start = gps_data[['pos_enu_x', 'pos_enu_y']].iloc[eval(settings.loc["outagestart"].Value)].to_numpy().astype(np.float)
            #     outage_pos_stop = gps_data[['pos_enu_x', 'pos_enu_y']].iloc[eval(settings.loc["outagestop"].Value)].to_numpy().astype(np.float)
            #     plt.plot(outage_pos_start[0], outage_pos_start[1], 'y*', markersize = 8, label = 'Start Outage')
            #     plt.plot(outage_pos_stop[0], outage_pos_stop[1], 'r*', markersize = 8, label = 'Stop Outage')

            plt.plot(pos_kf[0,:],pos_kf[1,:], 'g', linewidth=3, label = 'KF pos')
            plt.scatter(gps_track[0,:],gps_track[1,:], linewidth=1, label = 'Pos from Camera')
            # plt.xlim(left, right)
            # plt.ylim(left, right)
    if frame == 'NED':

        if gps_track is None:
            plt.plot(pos_kf[0,:],pos_kf[1,:], 'g', linewidth=3, label = 'KF pos')

        else:

            plt.plot(pos_kf[1,0], pos_kf[0,0], 'gp', markersize = 8, label = 'Start KF')
            plt.plot(pos_kf[1,len(pos_kf[1,:])-1], pos_kf[0,len(pos_kf[0,:])-1], 'rp', markersize = 8, label = 'Stop KF')

            plt.plot(gps_track[1,0], gps_track[0,0], 'gP', markersize = 8, label = 'Start Gps')
            plt.plot(gps_track[1,len(gps_track[1,:])-1], gps_track[0,len(gps_track[0,:])-1], 'rP', markersize = 8, label = 'Stop Gps')

            # if settings.loc["gnss_outage"].Value == 'on':
            #     outage_pos_start = gps_data[['enu_x', 'enu_y', 'enu_z']].iloc[eval(settings.loc["outagestart"].Value)].to_numpy().astype(np.float)
            #     outage_pos_stop = gps_data[['enu_x', 'enu_y', 'enu_z']].iloc[eval(settings.loc["outagestop"].Value)].to_numpy().astype(np.float)
            #     plt.plot(outage_pos_start[1], outage_pos_start[0], 'y*', markersize = 8, label = 'Start Outage')
            #     plt.plot(outage_pos_stop[1], outage_pos_stop[0], 'r*', markersize = 8, label = 'Stop Outage')

            plt.plot(pos_kf[1,:],pos_kf[0,:], 'g', linewidth=3, label = 'KF pos')
            # plt.scatter(pos_kf[1,:],pos_kf[0,:], linewidth=1)
            plt.scatter(gps_track[1,:],gps_track[0,:], linewidth=1, label = 'Gps Track')

    plt.title('Position in ENU (Local Frame)', fontsize = 18, fontweight = 'bold' )
    plt.xlabel('Y', fontsize = 16)
    plt.ylabel('Z', fontsize = 16)
    plt.grid()
    plt.legend()


def plot_error(pos_kf, gps_track, imu_time, gps_time):

    # get xy coordinates from pos_kf
    kf_x = pos_kf[0,:]
    kf_y = pos_kf[1,:] 

    # make an interpolation function along each axis
    fx = interp1d(gps_time, gps_track[0,:], fill_value='extrapolate')
    fy = interp1d(gps_time, gps_track[1,:], fill_value='extrapolate')

    # apply interpolation function to imu time
    rfid_x = fx(imu_time)
    rfid_y = fy(imu_time)

    xerr = kf_x - rfid_x
    yerr = kf_y - rfid_y

    plt.figure(figsize=(12,9))
    plt.plot(imu_time,xerr, 'g', linewidth=3, label = 'x error')
    plt.plot(imu_time,yerr, 'b', linewidth=3, label = 'y error')

    plt.title('Error in Position', fontsize = 18, fontweight = 'bold' )
    plt.xlabel('Time (sec)', fontsize = 16)
    plt.ylabel('Error (m)', fontsize = 16)
    plt.grid()
    plt.legend()


def plot_orientation(orient_eul, ref_orient=None):

    plt.figure(figsize=(12,9))

    if ref_orient is not None:
        # x = np.linspace(0, len(orient_eul[0,:]), len(orient_eul[0,:]))
        # x = len(orient_eul[0,:])

        # orient_eul[orient_eul < -20] = orient_eul[orient_eul < -20] + 360 

        plt.plot(orient_eul[0,:], label = 'Roll Local', linewidth=3)
        # plt.plot(orient_eul[1,:], label = 'Pitch Local', linewidth=3)
        plt.plot(orient_eul[2,:], label = 'Yaw Local', linewidth=3)

        # plt.plot(ref_orient[0,:], label = 'Roll Reference', linewidth=3)
        plt.plot(ref_orient[1,:], label = 'Pitch Reference', linewidth=3)
        plt.plot(ref_orient[2,:], label = 'Yaw Reference', linewidth=3)

        plt.title('Orientation from IMU', fontsize = 18, fontweight = 'bold' )
        plt.xlabel('Time (sec)', fontsize = 16)
        plt.ylabel('Angle (deg)', fontsize = 16)
        plt.legend()
        plt.grid()
    
    else:
        x = np.linspace(0, len(orient_eul[0,:])*0.02, len(orient_eul[0,:]))

        # orient_eul[orient_eul < -20] = orient_eul[orient_eul < -20] + 360 

        plt.plot(x, orient_eul[0,:], label = 'Roll', linewidth=3)
        plt.plot(x, orient_eul[1,:], label = 'Pitch', linewidth=3)
        plt.plot(x, orient_eul[2,:], label = 'Yaw', linewidth=3)

        plt.title('Orientation from IMU', fontsize = 18, fontweight = 'bold' )
        plt.xlabel('Time (sec)', fontsize = 16)
        plt.ylabel('Angle (deg)', fontsize = 16)
        plt.legend()
        plt.grid()


if __name__ == '__main__':

    sol_filename = "./sensor_yaw_out.txt"
    gps_filename = "./sensor_yaw.csv"

    sol_data = pd.read_csv(sol_filename, sep="\\s+")
    gps_data = pd.read_csv(gps_filename)
    
    # gps_time = np.round(gps_data['t'].to_numpy(), 3)
    # imu_time = sol_data['Time(s)']
    # print(len(imu_time))
    pos_kf = sol_data[['Pos_x(m)', 'Pos_y(m)']].to_numpy().T
    # gps_track = gps_data[['pos_enu_x', 'pos_enu_y']].to_numpy().T
    gps_track = (gps_data[['PositionCm.x', 'PositionCm.y', 'PositionCm.z']].to_numpy().T ) / 100
    ref_orient = gps_data[['Orientation.w', 'Orientation.x', 'Orientation.y', 'Orientation.z']].to_numpy().T

    ref_orient = R.from_quat(ref_orient.T).as_euler('xyz', degrees=True).T

    # ref_orient = []
    # for idx, row in gps_data.iterrows():
    #     q = row[16:20].to_numpy()
        
    #     ref_orient.append(R.from_quat(q).as_euler('xyz', degrees=True).T)
        
    # ref_orient = np.array(ref_orient).T

    print(np.shape(ref_orient))

    # define rotations 
    Rz = R.from_euler('z', -180, degrees=True).as_matrix()
    Rx = R.from_euler('x', -90, degrees=True).as_matrix()
    R_pos = np.matmul(Rx, Rz)

    gps_track_trans = np.matmul(R_pos, gps_track)[:2, :]
    # gps_track_trans = np.delete(gps_track_trans, 1, 0)

    # print(gps_track_trans)

    plot_pos(pos_kf, gps_track_trans, frame='ENU')
    # plot_error(pos_kf, gps_track, imu_time, gps_time)
    # orient_nav = []

    # for idx, row in sol_data.iterrows():
    #     q = row[7:].to_numpy()
        
    #     orient_nav.append(np.rad2deg(mu.quaternion_to_euler(*q)))
        
    # orient_nav = np.array(orient_nav).T
    
    # plot orientation
    orient = np.rad2deg(sol_data[['Roll(rad)', 'Pitch(rad)', 'Yaw(rad)']].to_numpy().T)
    print(np.shape(orient))
    plot_orientation(orient, ref_orient=ref_orient)

    # plot_animation(imu_time, gps_time, pos_kf, gps_track)

    plt.show()
