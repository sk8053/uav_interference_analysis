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
def get_df(dir_=None, n = 1, ground = True, f = 28, ISD = 200, n_t=8, standard_only = False):
    df = pd.DataFrame()
    t_n = np.array([])
    for t in range(n_iter):
        if standard_only is True:
            data = pd.read_csv('%s/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % (
                dir_, n, height, f, t), delimiter='\t')
        else:
            data = pd.read_csv('%s/uplink_itf_ISD=%d_ns=%d_h=%d_%dG_%d.txt' % (
                dir_,ISD, n,height, f, t), delimiter='\t')
        df = pd.concat([df,data])

    #df_uav = df[df['ue_type'] == 'uav']
    print(dir_, np.sum(df['ue_type']=='uav')/len(df), ISD)
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
    capacity = (n/n_t) * BW*np.log2(1 + 10**(0.1*SINR))
    return  np.array(SINR), SNR, 10*np.log10(total_itf_lin) - noise_power_dB, df['tx_power'], capacity

n_s = 1

dir_8 = ab_path+'/test_data/dedicated_1_1_ptrl/UAV_8'
dir_3 = ab_path+'/test_data/dedicated_1_1_ptrl/UAV_3'
dir_6 = ab_path+'/test_data/dedicated_1_1_ptrl/UAV_6'
dir_0 = ab_path + '/test_data/2_stream_ptrl/UAV_8'

df1_sinr_200_uav, df1_snr_200_uav, df1_inr_200_uav, tx_power_200_uav, capacity_200_uav \
    = get_df(dir_ = dir_0, n = 2, f= 28, ISD = 200, ground= False, n_t = 16, standard_only = True)
df1_sinr_200, df1_snr_200, df1_inr_200, tx_power_200, capacity_200 \
    = get_df(dir_ = dir_0, n = 2, f= 28, ISD = 200,  n_t = 16, standard_only = True)

df1_sinr_200_uav_8, df1_snr_200_uav_8, df1_inr_200_uav_8, tx_power_200_uav_8, capacity_200_uav_8 \
    = get_df(dir_ = dir_8, n = 1, f= 28, ISD = 200, ground= False, n_t = 8)
df1_sinr_200_gue_8, df1_snr_200_gue_8, df1_inr_200_gue_8, tx_power_200_gue_8, capacity_200_gue_8 \
    = get_df(dir_ = dir_8, n = 1, f= 28, ISD = 200,  n_t = 8)

df1_sinr_200_gue_6, df1_snr_200_gue_6, df1_inr_200_gue_6, tx_power_200_gue_6, capacity_200_gue_6 = \
    get_df(dir_ = dir_6, n = 1, f= 28, ISD = 200, n_t = 10)
df1_sinr_200_uav_6, df1_snr_200_uav_6, df1_inr_200_uav_6, tx_power_200_uav_6, capacity_200_uav_6 = \
    get_df(dir_ = dir_6, n = 1, f= 28, ISD = 200, ground=False, n_t = 6)

df1_sinr_200_gue_3, df1_snr_200_gue_3, df1_inr_200_gue_3, tx_power_200_gue_3, capacity_200_gue_3 = \
    get_df(dir_ = dir_3, n = 1, f= 28, ISD = 200, n_t = 13)
df1_sinr_200_uav_3, df1_snr_200_uav_3, df1_inr_200_uav_3, tx_power_200_uav_3, capacity_200_uav_3 = \
    get_df(dir_ = dir_3, n = 1, f= 28, ISD = 200, ground=False, n_t = 3)

#df1_sinr_400, df1_snr_400, df1_inr_400, tx_power_400, capacity_400 = get_df(dir_ = dir_, n = 1, f= 28, ISD = 400, n_t = 3)
#df1_sinr_800, df1_snr_800, df1_inr_800, tx_power_800 = get_df(dir_ = dir_, n = 1, f= 28, ISD = 800)


#df1_sinr_400_uav, df1_snr_400_uav, df1_inr_400_uav, tx_power_400_uav, capacity_400_uav = \
#    get_df(dir_ = dir_, n = 1, f= 28, ISD = 400, ground = False, n_t = 3)
#df1_sinr_800_uav, df1_snr_800_uav, df1_inr_800_uav, tx_power_800_uav = get_df(dir_ = dir_, n = 1, f= 28, ISD = 800, ground = False)

feature = "SINR"
fig, ax = plt.subplots(figsize = (8,6))
lines =[]
UE = 'gUE'
if feature == 'SINR':
    if UE == 'UAV':
        plt.plot(np.sort(df1_sinr_200_uav_8), np.linspace(0, 1, len(capacity_200_uav_8)), 'r', label='UAV 50%,'+ r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(df1_sinr_200_uav_6), np.linspace(0, 1, len(capacity_200_uav_6)), 'g', label='UAV 37%,' + r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(df1_sinr_200_uav_3), np.linspace(0, 1, len(capacity_200_uav_3)), 'b', label='UAV 18%,' + r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(df1_sinr_200_uav), np.linspace(0, 1, len(capacity_200_uav)), 'k',
                 label='no dedicated BS,' + r' ISD$_d$ = 200m', lw=2)
    else:
        plt.plot(np.sort(df1_sinr_200_gue_8), np.linspace(0, 1, len(df1_sinr_200_gue_8)), 'r', label='UAV 50%,'+ r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(df1_sinr_200_gue_6), np.linspace(0, 1, len(df1_sinr_200_gue_6)), 'g', label='UAV 37%,' + r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(df1_sinr_200_gue_3), np.linspace(0, 1, len(df1_sinr_200_gue_3)), 'b', label='UAV 18%,' + r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(df1_sinr_200), np.linspace(0, 1, len(df1_sinr_200)), 'k',
                 label='no dedicated BS,' + r' ISD$_d$ = 200m', lw=2)
if feature == 'C':
    if UE == 'UAV':
        plt.plot(np.sort(capacity_200_uav_8), np.linspace(0, 1, len(capacity_200_uav_8)), 'r', label='UAV 50%,'+ r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(capacity_200_uav_6), np.linspace(0, 1, len(capacity_200_uav_6)), 'g', label='UAV 37%,' + r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(capacity_200_uav_3), np.linspace(0, 1, len(capacity_200_uav_3)), 'b', label='UAV 18%,' + r' ISD$_d$ = 200m', lw=2)
        plt.plot(np.sort(capacity_200_uav), np.linspace(0, 1, len(capacity_200_uav)), 'k',
                 label='no dedicated BS,' + r' ISD$_d$ = 200m', lw=2)

    #    plt.plot(np.sort(df1_sinr_400_uav), np.linspace(0, 1, len(df1_sinr_400_uav)), 'g', label='ISD = 400m', lw=2)
 #       plt.plot(np.sort(df1_sinr_800_uav), np.linspace(0, 1, len(df1_sinr_800_uav)), 'k', label='ISD = 800m', lw=2)
    else:
        plt.plot(np.sort(df1_sinr_200), np.linspace(0, 1, len(df1_sinr_200)), 'r', label='ISD = 200m', lw=2)
        #plt.plot(np.sort(df1_sinr_200_8), np.linspace(0, 1, len(df1_sinr_200_uav_8)), 'r:', label='ISD = 200m', lw=2)
        plt.plot(np.sort(df1_sinr_400), np.linspace(0, 1, len(df1_sinr_400)), 'g', label='ISD = 400m', lw=2)
  #      plt.plot(np.sort(df1_sinr_800), np.linspace(0, 1, len(df1_sinr_800)), 'k', label='ISD = 800m', lw=2)

if feature == 'SNR':
    if UE == 'UAV':
        plt.plot(np.sort(df1_snr_200_uav), np.linspace(0, 1, len(df1_snr_200_uav)), 'r', label='ISD = 200m', lw=2)
        plt.plot(np.sort(df1_snr_400_uav), np.linspace(0, 1, len(df1_snr_400_uav)), 'g', label='ISD = 400m', lw=2)
  #      plt.plot(np.sort(df1_snr_800_uav), np.linspace(0, 1, len(df1_snr_800_uav)), 'k', label='ISD = 800m', lw=2)
    else:
        plt.plot(np.sort(df1_snr_200), np.linspace(0, 1, len(df1_snr_200)), 'r', label='ISD = 200m', lw=2)
        plt.plot(np.sort(df1_snr_400), np.linspace(0, 1, len(df1_snr_400)), 'g', label='ISD = 400m', lw=2)
   #     plt.plot(np.sort(df1_snr_800), np.linspace(0, 1, len(df1_snr_800)), 'k', label='ISD = 800m', lw=2)

if feature == 'INR':
    if UE == 'UAV':
        plt.plot(np.sort(df1_inr_200_uav), np.linspace(0, 1, len(df1_inr_200_uav)), 'r', label='ISD = 200m', lw=2)
        plt.plot(np.sort(df1_inr_400_uav), np.linspace(0, 1, len(df1_inr_400_uav)), 'g', label='ISD = 400m', lw=2)
    #    plt.plot(np.sort(df1_inr_800_uav), np.linspace(0, 1, len(df1_inr_800_uav)), 'k', label='ISD = 800m', lw=2)

    else:
        plt.plot(np.sort(df1_inr_200), np.linspace(0, 1, len(df1_inr_200)), 'r', label='ISD = 200m', lw=2)
        plt.plot(np.sort(df1_inr_400), np.linspace(0, 1, len(df1_inr_400)), 'g', label='ISD = 400m', lw=2)
     #   plt.plot(np.sort(df1_inr_800), np.linspace(0, 1, len(df1_inr_800)), 'k', label='ISD = 800m', lw=2)

if feature == 'Power':
    if UE == 'UAV':
        plt.plot(np.sort(tx_power_200_uav), np.linspace(0, 1, len(df1_inr_200_uav)), 'r', label='ISD = 200m', lw=2)
        plt.plot(np.sort(tx_power_400_uav), np.linspace(0, 1, len(df1_inr_400_uav)), 'g', label='ISD = 400m', lw=2)
    #    plt.plot(np.sort(tx_power_800_uav), np.linspace(0, 1, len(df1_inr_800_uav)), 'k', label='ISD = 800m', lw=2)

    else:
        plt.plot(np.sort(tx_power_200), np.linspace(0, 1, len(df1_inr_200)), 'r', label='ISD = 200m', lw=2)
        plt.plot(np.sort(tx_power_400), np.linspace(0, 1, len(df1_inr_400)), 'g', label='ISD = 400m', lw=2)
    #    plt.plot(np.sort(tx_power_800), np.linspace(0, 1, len(df1_inr_800)), 'k', label='ISD = 800m', lw=2)


    #plt.plot(np.sort(df2_inr_uav), np.linspace(0, 1, len(df2_inr_uav)), 'r:', label='case 2, UAV INR', lw=2)
#plt.title(r'CASE 1: BS$_s$' + u"\u2192" +'2 UE and 0 UAV'+ '\n'
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
if feature == 'C':
    plt.xlabel(r' Capacity ', fontsize=16)


#ax.legend(fontsize = 16)
#plt.xlim([-20,35])
#plt.ylabel('Probability of Coverage, P', fontsize = 16)
plt.ylabel('CDF ', fontsize = 16)

plt.grid()

if feature == 'SINR':
    plt.savefig('sub6_sinr_stream=%d_height=%dm_ptrl.png'%(n_s, height))
if feature == 'SNR':
    plt.savefig('sub6_snr_stream=%d_height=%dm_ptrl.png' % (n_s, height))
if feature == 'INR':
    plt.savefig('sub6_inr_stream=%d_height=%dm_ptrl.png' % (n_s, height))
#if feature == 'INR':
#    plt.savefig('sub6_inr_stream=%d_height=%dm_ptrl.png' % (n_s, height))

plt.show()



