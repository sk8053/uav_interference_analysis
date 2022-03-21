# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:06:42 2021

@author: gangs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from pyarrow import csv 
import seaborn as sns

data1 = pd.read_csv('data_1_copy.csv')
data2 = pd.read_csv('data_2_copy.csv')

## x, y of first part
xx1,yy1 = data1['rx_x']/5, data1['rx_y']/5
xx1_int, yy1_int = np.array(xx1, dtype = int), np.array(yy1, dtype = int)
## x, y of second part
xx2, yy2 = data2['rx_x']/5, data2['rx_y']/5
xx2_int, yy2_int = np.array(xx2, dtype = int), np.array(yy2, dtype = int)
snr1 = data1['SNR']
snr2 = data2['SNR']

max_xx_int = int(max(max(xx2), max(xx1))) +1
min_xx_int = int(min(min(xx2), min(xx1))) -1

max_yy_int = int(max(max(yy1), max(yy2)))
min_yy_int = int(min(min(yy2), min(yy1)))

x_length = max_xx_int - min_xx_int
y_length = max_yy_int - min_yy_int
map_env = np.ones ((y_length, x_length)) * np.NaN
yy1_int -= min_yy_int
xx1_int -= min_xx_int
yy2_int -= min_yy_int
xx2_int -= min_xx_int
map_env[yy1_int-1, xx1_int-1] = snr1
map_env[yy2_int-1, xx2_int-1] = snr2
fig, ax = plt.subplots(1,1)
img = ax.imshow(np.flip(map_env, axis =0))
np.savetxt('snr_map.txt', map_env)
fig.colorbar(img)
ax.set_yticks(np.arange(map_env.shape[0], step = 30))
ax.set_yticklabels(np.arange(max_yy_int*5,min_yy_int*5,  step = -150))

ax.set_xticks(np.arange(map_env.shape[1], step = 40))
ax.set_xticklabels(np.arange(max_xx_int*5,min_xx_int*5,  step = -200))
plt.show()
#ax.set_yticks(np.arange(min_yy_int*5, max_yy_int*5, step= 50))

#s.heatmap(map_env)
'''
xx -= min(xx)
yy -= min(yy)

max_int_yy = int(max(yy))
max_int_xx = int(max(xx))
map_1 = np.ones((max_int_yy, max_int_xx))*250
int_yy = np.array(yy, dtype = int)
int_xx = np.array(xx, dtype = int)
map_1[max_int_yy - int_yy-1, int_xx-1] = pl

sns.heatmap(map_1)
#plt.imshow(map_1)
'''