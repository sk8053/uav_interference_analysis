import random

import sys
import pathlib
import numpy as np
ab_path = pathlib.Path().absolute().parent.__str__()
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit


def p_los(dist2d, d1, p1):
    #p1 = 233.98 * np.log10(height) - 0.95
    #d1 = np.max(294.05 * np.log10(height) - 432.94, 18)
    los_prob = np.zeros(len(dist2d))
    los_prob[dist2d < d1] =1.0
    dist2d_copy = dist2d[dist2d>=d1]
    los_prob[dist2d >= d1]  =  (d1/dist2d_copy) + np.exp(-dist2d_copy/p1)*(1- d1/dist2d_copy)

    return los_prob

sys.path.append(ab_path+'/uav_interference_analysis/src/')

from src.helper.network_channel import Channel_Info, Network

channel_info = Channel_Info(city='uav_moscow')
channel = channel_info.aerial_channel

resolution = 2000

x_list = np.linspace(0, 500, resolution)
z_list = np.repeat([120], resolution)#np.flip(np.linspace(22.5, 150, resolution))

y_list = np.repeat([0], len(x_list))
dist_vect = np.column_stack((x_list, y_list, z_list))

cell_type_k = np.repeat([1], len(dist_vect))
ls_prob_list = []

for i in tqdm(range(500)):
    _, ls_list = channel.sample_path(dist_vect, cell_type_k)
    ls_list[ls_list ==2] = 0
    ls_prob_list.append(ls_list)

ls_prob_list = np.array(ls_prob_list)
ls_prob_list = np.mean(ls_prob_list, axis = 0)

dist_2d = np.abs(x_list)

print(dist_2d.shape, ls_prob_list.shape)
p_opt, p_cov = curve_fit(p_los, dist_2d, ls_prob_list)
print(p_opt)

los_prob_pred = p_los(dist_2d, *p_opt)

plt.scatter(dist_2d, ls_prob_list, s = 5)
plt.plot(dist_2d, los_prob_pred, 'r-.')
plt.show()
