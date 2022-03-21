# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:44:21 2022

@author: seongjoon kang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('drone_antenna_pattern.csv', header=None)

df = np.array(df)
df[:,0]-=90
df[:,1] -= 180

np.savetxt('drone_antenna_pattern_real_.csv', df)
ele = np.deg2rad(df[:,0])
az = np.deg2rad(df[:,1])
gain =10**(0.1*df[:,2])

r = gain
x = r*np.sin(ele)*np.cos(az)
y = r*np.sin(ele)*np.sin(az)
z = r* np.cos(ele)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot(x,y,z)
plt.show()