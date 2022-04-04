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

        data = pd.read_csv('%s/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % (
                dir_,n,height, f, t), delimiter='\t')
        df = pd.concat([df,data])

    print (np.sum(df['ue_type']=='g_ue')/len(df))
    if ground is True:
        df = df[df['ue_type']=='g_ue']
    else:
        df = df[df['ue_type']== 'uav']

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
dir_1 = ab_path+'/test_data/%d_stream_ptrl/UAV_0'%n_s
dir_2 = ab_path+'/test_data/%d_stream_ptrl/UAV_3'%n_s
dir_3 = ab_path+'/test_data/%d_stream_ptrl/UAV_6'%n_s
dir_4 = ab_path+'/test_data/%d_stream_ptrl/UAV_8'%n_s
#dir_2 = ab_path+'/test_data/2_stream/UE_4_UAV_4'

#dir_3 = 'new_data/%d_stream/%dm_height/UE_6_UAV_2' % (n_s, height)
#dir_4 = 'new_data/%d_stream/%dm_height/UE_5_UAV_3' % (n_s, height)
#dir_5 = 'new_data/%d_stream/%dm_height/UE_4_UAV_4' % (n_s, height)
#dir_5 = 'new_data/%d_stream/%dm_height/dedicated_2_2' % (n_s, height)

df1_sinr, df1_snr, df1_inr = get_df(dir_ = dir_1, n = n_s, f= 28)
df1_sinr_uav, df1_snr_uav, df1_inr_uav = get_df(dir_ = dir_1, n = n_s, ground= False, f = 28)

df2_sinr,df2_snr, df2_inr= get_df(dir_ = dir_2, n= 2, f= 28)
df2_sinr_uav,df2_snr_uav, df2_inr_uav= get_df(dir_ = dir_2, n= 2, f= 28, ground= False)

df3_sinr,df3_snr, df3_inr= get_df(dir_ = dir_3, n= n_s, f= 28)
df3_sinr_uav,df3_snr_uav, df3_inr_uav= get_df(dir_ = dir_3, n= n_s, f= 28, ground = False)

df4_sinr,df4_snr, df4_inr= get_df(dir_ = dir_4, n= n_s, f= 28)
df4_sinr_uav,df4_snr_uav, df4_inr_uav= get_df(dir_ = dir_4, n= n_s, f= 28, ground = False)

feature = "INR"
ue_type = "gUE"
fig, ax = plt.subplots(figsize = (8,6))
lines =[]
if feature == 'SINR':
    if ue_type == 'gUE':
        plt.plot(np.sort(df1_sinr), np.linspace(0,1,len(df1_sinr)),'k', label = '0% UAV', lw = 2)
        plt.plot(np.sort(df2_sinr), np.linspace(0,1,len(df2_sinr)),'r', label = '18.8% UAV', lw = 2)
        plt.plot(np.sort(df3_sinr), np.linspace(0, 1, len(df3_sinr)), 'g', label='37.5% UAV', lw=2)
        plt.plot(np.sort(df4_sinr), np.linspace(0, 1, len(df4_sinr)), 'b', label='50% UAV', lw=2)
    else:
        plt.plot(np.sort(df1_sinr_uav), np.linspace(0, 1, len(df1_sinr_uav)), 'k', label='0% UAV', lw=2)
        plt.plot(np.sort(df2_sinr_uav), np.linspace(0, 1, len(df2_sinr_uav)), 'r', label='18.8% UAV', lw=2)
        plt.plot(np.sort(df3_sinr_uav), np.linspace(0, 1, len(df3_sinr_uav)), 'g', label='37.5% UAV', lw=2)
        plt.plot(np.sort(df4_sinr_uav), np.linspace(0, 1, len(df4_sinr_uav)), 'b', label='50% UAV', lw=2)

if feature == 'SNR':
    if ue_type == 'gUE':
        plt.plot(np.sort(df1_snr), np.linspace(0,1,len(df1_sinr)),'k', label = '0% UAV', lw = 2)
        plt.plot(np.sort(df2_snr), np.linspace(0,1,len(df2_sinr)),'r', label = '18.8% UAV', lw = 2)
        plt.plot(np.sort(df3_snr), np.linspace(0, 1, len(df3_sinr)), 'g', label='37.5% UAV', lw=2)
        plt.plot(np.sort(df4_snr), np.linspace(0, 1, len(df4_sinr)), 'b', label='50% UAV', lw=2)
    else:
        plt.plot(np.sort(df1_snr_uav), np.linspace(0, 1, len(df1_sinr_uav)), 'k', label='0%, UAV', lw=2)
        plt.plot(np.sort(df2_snr_uav), np.linspace(0,1,len(df2_sinr_uav)),'r', label = '18.8%, UAV', lw = 2)
        plt.plot(np.sort(df3_snr_uav), np.linspace(0, 1, len(df3_sinr_uav)), 'g', label='37.5%, UAV', lw=2)
        plt.plot(np.sort(df4_snr_uav), np.linspace(0, 1, len(df4_sinr_uav)), 'b', label='50%, UAV', lw=2)

if feature == 'INR':
    if ue_type == 'gUE':
        plt.plot(np.sort(df1_inr), np.linspace(0,1,len(df1_sinr)),'k', label = '0% UAV', lw = 2)
        plt.plot(np.sort(df2_inr), np.linspace(0,1,len(df2_sinr)),'r', label = '18.8% UAV', lw = 2)
        plt.plot(np.sort(df3_inr), np.linspace(0, 1, len(df3_sinr)), 'g', label='37.5% UAV', lw=2)
        plt.plot(np.sort(df4_inr), np.linspace(0, 1, len(df4_sinr)), 'b', label='50% UAV', lw=2)
    else:
        plt.plot(np.sort(df1_inr_uav), np.linspace(0, 1, len(df1_sinr_uav)), 'k', label='0% UAV', lw=2)
        plt.plot(np.sort(df2_inr_uav), np.linspace(0, 1, len(df2_sinr_uav)), 'r', label='18.8% UAV', lw=2)
        plt.plot(np.sort(df3_inr_uav), np.linspace(0, 1, len(df3_sinr_uav)), 'g', label='37.5% UAV', lw=2)
        plt.plot(np.sort(df4_inr_uav), np.linspace(0, 1, len(df4_sinr_uav)), 'b', label='50% UAV', lw=2)
#plt.title(r'CASE 1: BS$_s$' + u"\u2192" +'1 UE and 1 UAV'+ '\n'
#          + r'CASE 2: BS$_s$' +u"\u2192" +r'1 UE and BS$_d$'+ u"\u2192" +'1 UAV', fontsize = 16)
legend1 = ax.legend(fontsize =16)
#legend2 = Legend(ax, [lines[0], lines[2]],['SINR', 'SNR'], fontsize = 14,
#                 bbox_to_anchor=(-0.02,1.115), loc="upper left", ncol = 2, frameon= False)
#ax.add_artist(legend2)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

if feature == 'SINR':
    plt.xlabel (r' SINR (dB)', fontsize = 16)
if feature == 'SNR':
    plt.xlabel(r' SNR (dB)', fontsize=16)
if feature == 'INR':
    plt.xlabel(r' INR (dB)', fontsize=16)


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



