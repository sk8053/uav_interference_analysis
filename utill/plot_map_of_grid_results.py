# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:06:42 2021
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from pyarrow import csv 
import seaborn as sns

data = pd.read_csv('snr_data_no_MIMO.csv')
snr = data['SNR']
x,y = data['x']/2, data['y']/2

x, y = np.array(x, dtype = int), np.array(y, dtype = int)

max_x = int(max(x))
min_x = int(min(x))

max_y = int(max(y))
min_y = int(min(y))


x_length = max_x - min_x + 1

y_length = max_y - min_y + 1
map_env = np.ones ((y_length, x_length)) * np.NaN

y -= min_y
x -= min_x

#snr[snr<-40] = np.NaN

map_env[y, x] = snr
fig, ax = plt.subplots(1,1)
img = ax.imshow(np.flip(map_env, axis =0))
np.savetxt('snr_map.txt', map_env)
fig.colorbar(img)
#print (max_x*2, max_y*2)
#print(min_x*2, min_y*2)

ax.set_yticks(np.arange(map_env.shape[0], step = 30))
ax.set_yticklabels(np.arange(max_y*2,min_y*2,  step = -2*30))

ax.set_xticks(np.arange(map_env.shape[1], step = 30))
ax.set_xticklabels(np.arange(min_x*2,max_x*2,  step = 30*2))

plt.show()
