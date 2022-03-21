import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n',action='store',default=30,type= int,\
    help='number of iteration')
parser.add_argument('--f',action='store',default=28e9,type= float,\
    help='frequency')

args = parser.parse_args()
n_iter = args.n
freq = args.f


ISD = 200
#ratio = 50
#n_UAV = 60

UE_power = 23
KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6

KT_lin = 10**(0.1*KT)
NF_lin = 10**(0.1*NF)
def get_df(UAV_Height, dir_=None):
    df = pd.DataFrame()
    t_n = np.array([])
    for t in range(n_iter):
        if freq == 28e9:
            data = pd.read_csv('../%s/data_%dm_height/uplink_interf_and_data_28G_%d.txt' % (
                dir_, UAV_Height,  t), delimiter='\t')
            t_n = np.append(t_n, np.repeat([len(data)/2-1], len(data[data['ue_type']=='uav'])))
        else:
            data = pd.read_csv('../data_with_3gpp_channel/data_%dm_height/uplink_interf_and_data_2G_%d.txt' % (
                UAV_Height, t), delimiter='\t')
        df = pd.concat([df,data])

    df_gue = df[df['ue_type'] == 'g_ue']

    #n_los = df_gue['n_los']
    #n_nlos = df_gue['n_nlos']
    #los_prob = n_los / (n_nlos + n_los)
    noise_power = KT_lin * NF_lin * BW
    return  df_gue['itf_UAV'] - 10*np.log10(noise_power)

dir_1 = 'data_with_nn_1_tilt_-12'
dir_2 = 'data_with_nn_2_tilt_-12'


n_los_120 = get_df(120, dir_ = dir_1)
n_los_30 = get_df(30, dir_ = dir_1)
n_los_60 = get_df(60, dir_ = dir_1)
n_los_90 = get_df(90, dir_ = dir_1)

n_los_120_2 = get_df(120, dir_ = dir_2)
n_los_30_2 = get_df(30, dir_ = dir_2)
n_los_60_2 = get_df(60, dir_ = dir_2)
n_los_90_2 = get_df(90, dir_ = dir_2)

plt.rcParams["font.family"] = "Times New Roman"
#fig, ax = plt.subplots(1,2)
#plt.figure(figsize=(7.5,5))
plt.subplot(1,2,1)
plt.plot(np.sort(n_los_120), np.linspace(0,1,len(n_los_120)),'r', label = '120m', lw = 2)
plt.plot(np.sort(n_los_90), np.linspace(0, 1, len(n_los_90)), 'b-.', label='90m', lw=2)
plt.plot(np.sort(n_los_60), np.linspace(0, 1, len(n_los_60)), 'g:', label='60m', lw=2)
plt.plot(np.sort(n_los_30), np.linspace(0,1,len(n_los_30)),'k--', label = '30m', lw = 2)

plt.legend(fontsize =15, loc = 'lower right')
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title ('1-connection', fontsize = 15, fontweight = '500')
plt.ylabel('CDF', fontsize = 15, fontweight = '500')


plt.grid()
plt.subplot(1,2,2)
plt.plot(np.sort(n_los_120_2), np.linspace(0,1,len(n_los_120_2)),'r', label = '120m', lw = 2)
plt.plot(np.sort(n_los_90_2), np.linspace(0, 1, len(n_los_90_2)), 'b-.', label='90m', lw=2)
plt.plot(np.sort(n_los_60_2), np.linspace(0, 1, len(n_los_60_2)), 'g:', label='60m', lw=2)
plt.plot(np.sort(n_los_30_2), np.linspace(0,1,len(n_los_30_2)),'k--', label = '30m', lw = 2)

plt.legend(fontsize =15, loc = 'lower right')
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('2-connection', fontsize  =15, fontweight = '500')
#plt.xlabel ('Fraction of LOS interfering links', fontsize = 13)

plt.grid()

#plt.suptitle('Fraction of LOS interfering links', x=0.5, y= 0.05, fontsize=15, fontweight = '500')
plt.suptitle('INR (dB)', x=0.5, y= 0.05, fontsize=15, fontweight = '500')

plt.show()
#plt.savefig('los_prob_different_heights.png', dpi = 1200)




