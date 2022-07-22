import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import sys
import pathlib
ab_path = pathlib.Path().absolute().parent.__str__()
print (ab_path)
sys.path.append(ab_path+'/uav_interference_analysis/test_data')

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n',action='store',default=30,type= int,\
    help='number of iteration')
parser.add_argument('--n_s',action='store',default=1,type= int,\
    help='number of streams')
parser.add_argument('--h',action='store',default=60,type= int,\
    help='UAV height')
parser.add_argument('--f',action='store',default=28e9,type= float,\
    help='frequency')

plt.rcParams["font.family"] = "Times New Roman"
args = parser.parse_args()
n_iter = args.n
freq = args.f
height = args.h
n_s= args.n_s


KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6

KT_lin = 10**(0.1*KT)
NF_lin = 10**(0.1*NF)
power_control = False
print (height)
def get_df(dir_=None, n = 1, ground = True, f = 28):
    df = pd.DataFrame()
    t_n = np.array([])
    for t in range(n_iter):

        data = pd.read_csv('%s/uplink_itf_UAV=5_ISD_d=200_ns=%d_h=%d_%dG_%d.txt' % (
                dir_,n,height, f, t), delimiter='\t')
        df = pd.concat([df,data])

    #print (np.sum(df['ue_type']=='g_ue')/len(df))
    #if ground is True:
    #    df = df[df['ue_type']=='g_ue']
    #else:
    #    df = df[df['ue_type']== 'uav']

    noise_power_dB = 10*np.log10(BW) + KT+NF
    noise_power_lin = KT_lin * NF_lin * BW

    #print (df_uav['tx_power'])
    tx_power = df['tx_power']

    l_f = df['l_f']
    intra_itf = df['intra_itf']
    inter_itf = df['inter_itf']

    if n ==1:
        intra_itf_lin = 0
    else:
        intra_itf_lin = 10 ** (0.1 * intra_itf)

    inter_itf_lin = 10**(0.1*inter_itf)
    total_itf_lin = intra_itf_lin +  inter_itf_lin
    noise_and_itf_dB = 10*np.log10(noise_power_lin + total_itf_lin)

    SINR =tx_power + l_f - noise_and_itf_dB
    SNR = tx_power + l_f - noise_power_dB
    #p = 0.5
    #med_ind = int(len(SINR)*p)
    #median = np.sort(SINR)[med_ind]
    #median_SNR = np.sort(SNR)[med_ind]
    #print (dir_[-10:], median, median_SNR)
    #SNR = df['bs_elem']
    return  np.array(SINR), SNR, 10*np.log10(total_itf_lin) - noise_power_dB,

n_s = 2

dir_2 = ab_path+'/test_data/%d_stream_ptrl'%n_s

df2_sinr,df2_snr, df2_inr= get_df(dir_ = dir_2, n= 2, f= 28)
#df2_sinr_uav,df2_snr_uav, df2_inr_uav= get_df(dir_ = dir_2, n= 2, f= 28, ground= False)

#df3_sinr,df3_snr, df3_inr= get_df(dir_ = dir_3, n= n_s, f= 28)
#df3_sinr_uav,df3_snr_uav, df3_inr_uav= get_df(dir_ = dir_3, n= n_s, f= 28, ground = False)

#df4_sinr,df4_snr, df4_inr= get_df(dir_ = dir_4, n= n_s, f= 28)
#df4_sinr_uav,df4_snr_uav, df4_inr_uav= get_df(dir_ = dir_4, n= n_s, f= 28, ground = False)

feature = "INR"
ue_type = "gUE"
fig, ax = plt.subplots(figsize = (8,6))
lines =[]
plt.plot(np.sort(df2_sinr), np.linspace(0,1,len(df2_sinr)),'r', label = 'MMSE beamforming', lw = 2)


#ax.legend(fontsize = 16)
plt.xlim([-40,40])
#plt.ylabel('Probability of Coverage, P', fontsize = 16)
plt.ylabel('CDF ', fontsize = 16)

plt.grid()
if ue_type == 'gUE':
    adds = 'g_ue_'
else:
    adds = 'uav_'
if feature == 'SINR':
    plt.savefig(adds+'sinr_%d-connections_height=%dm_ptrl.png'%(n_s, height))
if feature == 'SNR':
    plt.savefig(adds+'snr_%d-connections_height=%dm_ptrl.png' % (n_s, height))
if feature == 'INR':
    plt.savefig(adds+'inr_%d-connections_height=%dm_ptrl.png' % (n_s, height))
plt.show()



