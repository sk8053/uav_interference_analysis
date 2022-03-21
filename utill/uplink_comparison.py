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
def extract_data(data):
    #UE_type = data['UE_type']
    # extract SINR, SNR, and interference of UAV
    SINR = data['SINR']
    SNR = data['SNR']
    Itf = data['interference']
    ratio = data['ratio']
    Itf = Itf.replace([-np.inf], np.nan).dropna()
    Itf_lin = 10 ** (0.1 * Itf)


    KT_lin, NF_lin = 10 ** (0.1 * channel_info.KT), 10 ** (0.1 * channel_info.NF)
    Noise_Interference_uav = KT_lin * NF_lin * channel_info.BW + Itf_lin * ratio
    Noise_Interference_gue = KT_lin * NF_lin * channel_info.BW + Itf_lin * (1 - ratio)

    TX_PL = SNR + channel_info.KT + channel_info.NF + 10 * np.log10(channel_info.BW)
    SINR_uav = TX_PL - 10 * np.log10(Noise_Interference_uav)
    SINR_gue = TX_PL - 10 * np.log10(Noise_Interference_gue)
    #return SINR, SNR, Itf
    return SINR_uav, SINR_gue, Itf, Itf_lin*ratio, Itf_lin* (1-ratio)

data = dict()
data['30'] = pd.read_csv('../data/uplink_interf_and_data_30_gUE_only.txt', delimiter = '\t', index_col = False)
data['60'] = pd.read_csv('../data/uplink_interf_and_data_60_gUE_only.txt', delimiter = '\t', index_col = False)
data['120'] = pd.read_csv('../data/uplink_interf_and_data_120_gUE_only.txt', delimiter = '\t', index_col = False)
KT_lin, NF_lin = 10 ** (0.1 * channel_info.KT), 10 ** (0.1 * channel_info.NF)

colors = ['r', 'g','b']
for i, h in enumerate(['30','60','120']):
    SINR_uav, SINR_gue, Itf,Itf_uav, Itf_gue = extract_data(data[h])
    Itf_gue, Itf_uav = 10*np.log10(Itf_gue), 10*np.log10(Itf_uav)
    if itf is False:
        plt.plot(np.sort(SINR_uav), np.linspace(0,1,len(SINR_uav)),colors[i]+'-.', label = 'SINR '+ h+' m caused by UAV')
        plt.plot(np.sort(SINR_gue), np.linspace(0,1,len(SINR_gue)),colors[i]+'-', label = 'SINR '+ h +' m caused by ground UE')
        plt.title ('SINR comparison for each height')
        file_name = 'SINR comparison for all heights'
    else:
        plt.plot (np.sort(Itf_uav), np.linspace(0,1,len(Itf)), colors[i] +'-.',label = h+' m caused by UAV')
        plt.plot(np.sort(Itf_gue), np.linspace(0, 1, len(Itf)), colors[i]+'-', label=h + ' m caused by ground UE')
        plt.xlabel('Interference (dBm)', fontsize = 12)
        plt.title ('Interference Power (dBm)')
        file_name = 'Interference comparison for all heights'

plt.legend()
plt.grid()

plt.ylabel ('CDF', fontsize = 12)
plt.savefig("../plots/"+ file_name)
plt.show()
