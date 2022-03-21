import random

import sys
import pathlib

import matplotlib.pyplot as plt

ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path+'/uav_interference_analysis/src/')

import numpy as np
from tqdm.auto import tqdm
from ue import  UE
from bs import BS
from network_channel import Channel_Info, Network
import pandas as pd
from ground_channel_generation import GroundChannel
from mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
#import argparse
#from wrap_around import wrap_around
import seaborn as sns
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss_multi_sect
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
import tensorflow.keras.backend as K
from mmwchanmod.datasets.download import load_model

x = np.linspace(-200, 200, 50)
y = np.linspace (-200, 200, 50)
xx, yy = np.meshgrid(x,y)
xx, yy = xx.reshape(-1), yy.reshape(-1)
z = np.repeat([30], len(xx))
dist_v = np.column_stack((xx,yy,z))
dist3D = np.linalg.norm (dist_v, axis = 1)
bs_type = np.repeat([1], len(dist_v))
frequency = 28e9

element_ue = Elem3GPP(thetabw=65, phibw=65)
drone_antenna_gain = drone_antenna_gain()
ue_ant = URA(elem=element_ue, nant=np.array([4, 4]), fc=frequency, drone_antenna_gain=drone_antenna_gain)
arr_ue = RotatedArray(ue_ant, theta0=-90, drone = True)
arr_ue_list = [arr_ue]

elem_gnb_t = Elem3GPP(thetabw=65, phibw=65)
arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8, 8]), fc=frequency)
arr_gnb_list_t = multi_sect_array(arr_gnb_t, sect_type='azimuth', nsect=3, theta0= -12)

K.clear_session
channel = load_model('uav_boston', src='remote')
channel.load_link_model()
channel.load_path_model()
channels, link_states = channel.sample_path(dist_v, bs_type)
pl =[]
for j, channel in tqdm(enumerate(channels), total = len(channels)):
    data = dir_path_loss_multi_sect(arr_gnb_list_t, arr_ue_list, channel, dist3D[j],
                                            long_term_bf=True,
                                            isdrone=True, frequency= frequency)
    sect_id = data['sect_ind']
    tx_elem_gain = data['ue_elem_gain_dict'][sect_id]
    pl.append(max(tx_elem_gain))
pl = np.array(pl).reshape(50,50)
plt.imshow(pl)
plt.colorbar()
plt.show()