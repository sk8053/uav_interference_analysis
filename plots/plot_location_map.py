import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
UAV_Height = 30
ISD = 200
ratio = 50
n_UAV = 60

for t in range (1):
    data = pd.read_csv('../data_with_3gpp_channel/data_%dm_height/uplink_interf_and_data_28G_%d.txt' % (
            UAV_Height, t), delimiter='\t')

    bs_id = data['bs_id']
    ue_id = data['ue_id']
    bs_x, bs_y, bs_z = data['bs_x'], data['bs_y'], data['bs_z']
    ue_x, ue_y, ue_z = data['ue_x'], data['ue_y'],  data['ue_z']
    lf = data['l_f']
    bs_loc = np.column_stack((bs_x, bs_y, bs_z))
    ue_loc = np.column_stack((ue_x, ue_y, ue_z))
    dist_vect = bs_loc - ue_loc
    distance = np.linalg.norm(dist_vect, axis = 1)
    print (min(distance), max(lf))

n_bs = np.max(bs_id)
loc = {b_id:[] for b_id in bs_id}
for i, b_id in enumerate(bs_id):
    loc[b_id].append((ue_id[i], bs_x[i], bs_y[i], ue_x[i], ue_y[i]))
bs_id_uniq = np.unique(bs_id)
for b_id in bs_id_uniq:
    for j in range (len(loc[b_id])):
        u_id, bs_x, bs_y, ue_x, ue_y = loc[b_id][j]

        plt.scatter(bs_x, bs_y, color = 'k')
        if u_id > n_UAV:
            plt.scatter(ue_x, ue_y, color = 'b', marker = '*')
            plt.plot([bs_x, ue_x], [bs_y, ue_y], 'b-.', lw =0.5)
        else:
            if u_id == 1:
                plt.scatter(ue_x, ue_y, color='r', marker='*', label='UAV')
            else:
                plt.scatter(ue_x, ue_y, color='r', marker='*')
            plt.plot([bs_x, ue_x], [bs_y, ue_y], 'r-.', lw=0.5)


plt.scatter(ue_x, ue_y, color='b', marker='*', label='g_UE')
plt.scatter(bs_x, bs_y, color = 'k', label = 'BS')

plt.legend()
plt.grid()
plt.show()