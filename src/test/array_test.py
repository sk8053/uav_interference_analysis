## author: seongjoon kang 04/April, 2023

## this code is to test antenna element gain when arrays are rotated
## element gain of drone is from csv file
## element gain of UAV is from 3gpp specification, 38.901

import numpy as np
import sys;
sys.path.append('..')
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, multi_sect_array, RotatedArray
from mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
import matplotlib.pyplot as plt
thetabw = 65
phibw = 65
fc = 28e9
drone_test = False
n_phi = 360
n_theta = 180
n_sect = 3
aoa_phi_ = np.linspace(-180, 180, n_phi)
aoa_theta_ = np.linspace(-90, 90, n_theta)

## element gain test for drones
aoa_phi, aoa_theta = np.meshgrid(aoa_phi_, aoa_theta_)

if drone_test == True:
    drone_antenna_gain = drone_antenna_gain(path = '/home/nyu_wireless/uav_uplink_interference_analysis/uav_interference_analysis/src/mmwchanmod/sim/' )
    ant_elem = Elem3GPP(thetabw=thetabw, phibw=phibw)
    ant_array = URA(elem=ant_elem, nant=np.array([4, 4]), fc=fc, drone_antenna_gain=drone_antenna_gain)
    # rotation angles are inclination angles
    rot_ant_array = RotatedArray(ant_array, theta0= 180, phi0= 20 , drone = True)

    ue_elem_gaini = rot_ant_array.sv(aoa_phi.reshape(-1), aoa_theta.reshape(-1), return_elem_gain=True, drone=True)
    ue_elem_gaini = ue_elem_gaini.reshape(n_theta,n_phi)
    plt.figure()
    plt.imshow(ue_elem_gaini)
    plt.colorbar()
    I_th = np.arange(n_theta, step = 10)
    plt.yticks(I_th, np.round(aoa_theta_[I_th]))

    I_phi = np.arange(n_phi, step=30)
    plt.xticks(I_phi, np.round(aoa_phi_[I_phi]))
    plt.ylabel ('elevation angle')
    plt.xlabel('azimuth angle')
    plt.title('-90 degree downtilted antenna gain')
    plt.savefig('fig/element gain of drone with -45 downtilted antennas.png')

    plt.show()
else:
    ant_elem = Elem3GPP(thetabw=thetabw, phibw=phibw)
    ant_array = URA(elem=ant_elem, nant=np.array([8, 8]), fc=fc, drone_antenna_gain=drone_antenna_gain)
    arr_gnb_list_t = multi_sect_array(ant_array, sect_type='azimuth', nsect=n_sect, theta0=-12)

    #print(aoa_theta)
    bs_elem_gain  = None
    for i in range(n_sect):
        bs_elem_gaini = arr_gnb_list_t[i].sv(aoa_phi.reshape(-1), aoa_theta.reshape(-1), return_elem_gain=True)
        if i ==0:
            bs_elem_gain = bs_elem_gaini
        else:
            bs_elem_gain = np.maximum(bs_elem_gain, bs_elem_gaini)

    plt.figure()
    plt.imshow(bs_elem_gain.reshape(n_theta,n_phi))
    plt.colorbar()
    I_th = np.arange(n_theta, step = 10)
    plt.yticks(I_th, np.round(aoa_theta_[I_th]))

    I_phi = np.arange(n_phi, step=30)
    plt.xticks(I_phi, np.round(aoa_phi_[I_phi]))
    plt.ylabel ('elevation angle')
    plt.xlabel('azimuth angle')

    plt.title('-12 degree downtilted antenna gain\n number of sector = %d'% n_sect)

    plt.savefig('fig/element gain with 3 sectors with -12 downtilted antennas.png')
    plt.show()
###