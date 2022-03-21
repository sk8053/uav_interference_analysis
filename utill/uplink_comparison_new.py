import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import pathlib

ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path + '/src/')
from network_channel import Channel_Info

parser = argparse.ArgumentParser(description='')

parser.add_argument('--itf',default='False', action = 'store_true')
parser.set_defaults(itf=False)

args = parser.parse_args()
itf = bool(args.itf)
channel_info = Channel_Info()


data_gue = dict()
data_gue['30'] = pd.read_csv('../data/uplink_interf_and_data_30_gUE_only.txt', delimiter = '\t', index_col = False)
data_gue['60'] = pd.read_csv('../data/uplink_interf_and_data_60_gUE_only.txt', delimiter = '\t', index_col = False)
data_gue['120'] = pd.read_csv('../data/uplink_interf_and_data_120_gUE_only.txt', delimiter = '\t', index_col = False)
data_uav = dict()
data_uav['30'] = pd.read_csv('../data/uplink_interf_and_data_30_uav_only.txt', delimiter = '\t', index_col = False)
data_uav['60'] = pd.read_csv('../data/uplink_interf_and_data_60_uav_only.txt', delimiter = '\t', index_col = False)
data_uav['120'] = pd.read_csv('../data/uplink_interf_and_data_120_uav_only.txt', delimiter = '\t', index_col = False)

KT_lin, NF_lin = 10 ** (0.1 * channel_info.KT), 10 ** (0.1 * channel_info.NF)
noise = channel_info.KT + channel_info.NF + 10 * np.log10(channel_info.BW)
colors = ['r', 'g','b']
for i, h in enumerate(['30','60','120']):

    SINR_uav, Itf_uav = data_uav[h]['SINR'], data_uav[h]['interference']
    SINR_gue, Itf_gue = data_gue[h]['SINR'], data_gue[h]['interference']
    INR_uav = Itf_uav - noise
    INR_gue = Itf_gue - noise
    if itf is False:
        plt.plot(np.sort(SINR_uav), np.linspace(0,1,len(SINR_uav)),colors[i]+':',
                 label = 'SINR '+ h+' m with only UAV')
        plt.plot(np.sort(SINR_gue), np.linspace(0, 1, len(SINR_gue)), colors[i] + '-',
                 label='SINR ' + h + ' m with only gUE')

        plt.title ('SINR comparison ')

    else:
        plt.plot (np.sort(INR_uav), np.linspace(0,1,len(Itf_uav)), colors[i] +':',
                  label = h+' m with only UAV')
        plt.plot (np.sort(INR_gue), np.linspace(0,1,len(Itf_gue)), colors[i] +'-',
                  label = h+' m with only gUE')
        plt.xlabel('INR (dB)', fontsize = 12)
        plt.title ('INR (dB)')

plt.legend()
plt.grid()

plt.ylabel ('CDF', fontsize = 12)
plt.show()
