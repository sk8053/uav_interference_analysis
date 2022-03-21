import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib

ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path + '/src/')
#sys.path.append('../')
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='')
from network_channel import Channel_Info
channel_info = Channel_Info()
parser.add_argument('--itf',default='False', action = 'store_true')
parser.set_defaults(itf=False)
parser.add_argument('--height',action='store',default=60,type= int,help='uav_height')

args = parser.parse_args()
uav_height = args.height
itf = bool(args.itf)

data = pd.read_csv('../data/uplink_interf_and_data_' + str (uav_height)+'.txt', delimiter = '\t', index_col = False)

# extract SINR, SNR, and interference of UAV
SINR = data['SINR']
SNR = data['SNR']
Itf = data['interference']
ratio = data['ratio']
Itf_lin = 10**(0.1*Itf)

link_state = data['link_state']

KT_lin, NF_lin = 10 ** (0.1 * channel_info.KT), 10 ** (0.1 * channel_info.NF)
Noise_Interference_uav = KT_lin * NF_lin * channel_info.BW +Itf_lin *ratio
Noise_Interference_gue = KT_lin * NF_lin * channel_info.BW +Itf_lin *(1-ratio)

Itf_uav = 10*np.log10(Itf_lin *ratio)
Itf_gue = 10*np.log10(Itf_lin *(1-ratio))

TX_PL = SNR + channel_info.KT + channel_info.NF + 10*np.log10(channel_info.BW)
SINR_uav = TX_PL - 10*np.log10(Noise_Interference_uav)
SINR_gue = TX_PL - 10*np.log10(Noise_Interference_gue)

los_prob = len(link_state[link_state==1])/len(link_state)
Itf= Itf.replace([-np.inf], np.nan).dropna()

if itf is False:
    plt.plot(np.sort(SINR), np.linspace(0,1,len(SINR)),'r-.', label = 'SINR ')
    plt.plot(np.sort(SNR), np.linspace(0,1,len(SNR)),'k-', label = 'SNR ')
    plt.title ('SINR  comparison')
    file_name = 'SINR  comparison'
else:
    plt.plot (np.sort(Itf_uav), np.linspace(0,1,len(Itf_uav)), label = 'interference by UAV')
    plt.plot(np.sort(Itf_gue), np.linspace(0, 1, len(Itf_gue)), label='interference by ground UE')
    plt.title ('Interference Comparison')
    file_name = 'Interference comparison'

plt.legend()
plt.title('UAV Height = '+ str(uav_height)+'m' ) #, LOS probability = '+ str(np.round(los_prob,2)))
plt.grid()
plt.ylabel ('CDF')
plt.savefig("../plots/"+ file_name+ str(uav_height)+"m")
plt.show()
