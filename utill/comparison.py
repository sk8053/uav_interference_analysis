import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import pathlib

ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path + '/src/')

from network_channel import Channel_Info
channel_info = Channel_Info()

parser = argparse.ArgumentParser(description='')

parser.add_argument('--itf',default='False', action = 'store_true')
parser.add_argument('--p',default='False', action = 'store_true')
parser.set_defaults(itf=False)
parser.set_defaults(p=False)

args = parser.parse_args()
itf = bool(args.itf)
p = bool(args.p)

def extract_data(data):
    #UE_type = data['UE_type']
    # extract SINR, SNR, and interference of UAV
    SINR = data['SINR']
    SNR = data['SNR']
    Itf = data['interference']
    tx_power = data['UE_TX_Power']
    Itf = Itf.replace([-np.inf], np.nan).dropna()
    return SINR, SNR, Itf, tx_power

data = dict()
data['30'] = pd.read_csv('../data/uplink_power_control_0/uplink_interf_and_data_30.txt', delimiter = '\t', index_col = False)
data['60'] = pd.read_csv('../data/uplink_power_control_0/uplink_interf_and_data_60.txt', delimiter = '\t', index_col = False)
data['120'] = pd.read_csv('../data/uplink_power_control_0/uplink_interf_and_data_120.txt', delimiter = '\t', index_col = False)
KT_lin, NF_lin = 10 ** (0.1 * channel_info.KT), 10 ** (0.1 * channel_info.NF)
Noise = 10*np.log10( KT_lin * NF_lin * channel_info.BW )
colors = ['r', 'g','b']
for i, h in enumerate(['30','60','120']):
    SINR, SNR, Itf, tx_power = extract_data(data[h])
    if p is True:
        plt.plot (np.sort(tx_power), np.linspace(0,1,len(tx_power)), colors[i]+'-', label = 'Tx power '+h + ' m')
        plt.title('Transmit power comparison for all heights')
        file_name = 'Tx Power comparison'
    elif itf is True:
        plt.plot (np.sort(Itf), np.linspace(0,1,len(Itf)), colors[i],label = h+' m')
        plt.title ('Interference')
        file_name = 'Interference comparison for all heights'
    else:
        plt.plot(np.sort(SINR), np.linspace(0, 1, len(SINR)), colors[i] + '-.', label='SINR ' + h + ' m')
        plt.plot(np.sort(SNR), np.linspace(0, 1, len(SNR)), colors[i] + '-', label='SNR ' + h + ' m')
        plt.title('SINR and SNR comparison for each height')
        file_name = 'SINR and SNR comparison for all heights'

plt.legend()
plt.grid()

plt.ylabel ('CDF')
plt.savefig("../plots/"+ file_name)
plt.show()
