import random
#import matplotlib.pyplot as plt
#import cProfile
#import pstats
import sys
import pathlib

import matplotlib.pyplot as plt

ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path+'/uav_interference_analysis/src/')
#sys.path.append(ab_path+'/uav_interference_analysis/src/mmwchanmod/sim/')

import numpy as np
from tqdm.auto import tqdm
from ue import  UE
from bs import BS
from network_channel import Channel_Info, Network
import pandas as pd
import seaborn as sns
from ground_channel_generation import GroundChannel

xx = np.linspace(-500, 500, 60)
zz = np.linspace(10, 200, 60)
#zz = np.repeat([120], len(xx))
#yy = xx.copy()

xx_bin, zz_bin = np.meshgrid(xx, zz)
xx_bin, zz_bin = xx_bin.reshape(-1), zz_bin.reshape(-1)
yy_bin = np.zeros_like(zz_bin) #+ np.random.uniform(0,20,size = zz_bin.shape)
#zz_bin = np.repeat([120], len(xx_bin))
rx_dist = np.column_stack((xx_bin, yy_bin, zz_bin))
tx_dist = np.zeros_like(rx_dist)
tx_dist[:,2] = 10
frequency = 28e9
n_iter = 20
los_map_list = np.zeros((n_iter,len(xx)*len(zz)), dtype = float)
pl_list = np.zeros((n_iter, len(xx)*len(zz)))
for iter in tqdm(range(n_iter)):

    uav_channels = GroundChannel(tx=tx_dist, rx=rx_dist, aerial_fading=True, frequency=frequency)
    data = open(uav_channels.path + 'ray_tracing.txt', 'r')
    uav_channel_list = uav_channels.getList(data)
    uav_channel_list = np.array(uav_channel_list)

    los_map = np.zeros(len(zz)*len(xx))
    pl_s = []
    for i, ch in enumerate(uav_channel_list):
        if ch.link_state == 1:
            los_map[i] = 1.0
        else:
            los_map[i] =0.0
        pl_s.append(ch.pl[0])
    los_map_list[iter] = los_map
    pl_list[iter] = pl_s

los_map = np.mean(los_map_list, axis = 0, dtype = float)
los_map = los_map.reshape(len(zz),len(xx))
los_map = np.flip(los_map, axis = 0)
sb  = sns.heatmap(los_map, cmap = 'jet')

sb.set_yticks(np.arange(0, los_map.shape[0]+6, step=6))
sb.set_yticklabels(np.flip(np.arange(10, 200+14, step=20)), rotation=0, size=14)

sb.set_xticks(np.arange(0, los_map.shape[1]+6, step = 6))
sb.set_xticklabels(np.arange(-500, 500+100, step = 100), rotation = 45, size = 14)

sb.set_xlabel('X', size=14)
sb.set_ylabel('Z', size=14)
sb.set_title('LOS probability', size = 14)
plt.imshow(los_map)


plt.figure()
distance = np.linalg.norm(tx_dist-rx_dist, axis = 1)
#pl_list = pl_list.reshape(len(zz), len(xx))
#pl_list = np.flip(pl_list)
#plt.imshow(pl_list)
#plt.colorbar()
pl_f = 20 *np.log10(distance) + 20*np.log10(frequency) - 147.55
plt.scatter(distance, pl_list[0].reshape(-1))
plt.scatter(distance, pl_f,color = 'r')
plt.xlabel('Distance (m)')
plt.ylabel ('Path loss (dB)')
plt.show()
