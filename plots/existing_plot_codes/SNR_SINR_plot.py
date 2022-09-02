import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
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
def get_df(dir_=None, n = 1):
    df = pd.DataFrame()
    t_n = np.array([])
    for t in range(n_iter):

        data = pd.read_csv('../%s/uplink_interf_and_data_28G_%d.txt' % (
                dir_, t), delimiter='\t')
        df = pd.concat([df,data])

    #df_uav = df[df['ue_type'] == 'uav']

    df = df[df['ue_type']=='uav']

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

    SINR = tx_power + l_f - noise_and_itf_dB
    SNR = tx_power + l_f - noise_power_dB
    p = 0.5
    med_ind = int(len(SINR)*p)
    median = np.sort(SINR)[med_ind]
    median_SNR = np.sort(SNR)[med_ind]
    print (dir_[-10:], median, median_SNR)
    return  np.array(SINR), SNR

if power_control is True:
    dir_1 = 'new_data_ptrl/%d_stream/%dm_height/UE_8_UAV_0'%(n_s, height)
    dir_2 = 'new_data_ptrl/%d_stream/%dm_height/UE_7_UAV_1'%(n_s, height)
    dir_3 ='new_data_ptrl/%d_stream/%dm_height/UE_6_UAV_2'%(n_s, height)
    dir_4 ='new_data_ptrl/%d_stream/%dm_height/UE_5_UAV_3'%(n_s, height)
    dir_5 = 'new_data_ptrl/%d_stream/%dm_height/dedicated_2_2'%(n_s, height)
    #dir_5 = 'new_data_ptrl/%d_stream/%dm_height/UE_4_UAV_4' % (n_s, height)
else:
    dir_1 = 'new_data/%d_stream/%dm_height/UE_8_UAV_0' % (n_s, height)
    dir_2 = 'new_data/%d_stream/%dm_height/UE_7_UAV_1' % (n_s, height)
    dir_3 = 'new_data/%d_stream/%dm_height/UE_6_UAV_2' % (n_s, height)
    dir_4 = 'new_data/%d_stream/%dm_height/UE_5_UAV_3' % (n_s, height)
    #dir_5 = 'new_data/%d_stream/%dm_height/UE_4_UAV_4' % (n_s, height)
    dir_5 = 'new_data/%d_stream/%dm_height/dedicated_2_2' % (n_s, height)

#df1_sinr, df1_snr = get_df(dir_ = dir_1, n = n_s)
df2_sinr,df2_snr= get_df(dir_ = dir_2, n= n_s)
df3_sinr,df3_snr= get_df(dir_ = dir_3, n = n_s)
df4_sinr,df4_snr= get_df(dir_ = dir_4, n= n_s)
df5_sinr,df5_snr= get_df(dir_ = dir_5, n = n_s)

fig, ax = plt.subplots(figsize = (8,6))
lines =[]
if True:
    #lines += ax.plot(np.sort(df1_sinr), np.linspace(0,1,len(df1_sinr)),'b', label = 'UAV = 0 %', lw = 2)
    lines += ax.plot(np.sort(df2_sinr), np.linspace(0,1,len(df2_sinr)),'g', label = 'UAV = 12.5 %', lw = 2)
    #lines += ax.plot(np.sort(df3_sinr), np.linspace(0,1,len(df3_sinr)),'k', label = 'UAV = 25 %', lw =2)
    lines += ax.plot(np.sort(df4_sinr), np.linspace(0,1,len(df4_sinr)),'r', label = 'UAV = 37.5 %', lw=2)
    lines += ax.plot(np.sort(df5_sinr), np.linspace(0, 1, len(df5_sinr)), 'c', label='UAV = 50 %', lw=2)
if True:
    #lines+=  ax.plot(np.sort(df1_snr), np.linspace(0,1,len(df1_snr)),'b:', label = 'UAV = 0 %', lw = 2)
    lines += ax.plot(np.sort(df2_snr), np.linspace(0,1,len(df2_snr)),'g:', label = 'UAV = 12.5 %', lw = 2)
    #lines += ax.plot(np.sort(df3_snr), np.linspace(0,1,len(df3_snr)),'k:', label = 'UAV = 25 %', lw =2)
    lines += ax.plot(np.sort(df4_snr), np.linspace(0,1,len(df4_snr)),'r:', label = 'UAV = 37.5 %', lw=2)
    lines += ax.plot(np.sort(df5_snr), np.linspace(0, 1, len(df5_snr)), 'c:', label='UAV = 50 %', lw=2)



legend1 = ax.legend(fontsize =13)
legend2 = Legend(ax, [lines[2], lines[5]],['SINR', 'SNR'], fontsize = 13,
                 bbox_to_anchor=(-0.02,1.115), loc="upper left", ncol = 2, frameon= False)
ax.add_artist(legend2)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

#plt.xlabel('SINR and SNR (dB)', fontsize = 16)
#plt.xlabel (r' SINR and SNR Threshold T (dB), N$_s$ = %d'% n_s, fontsize = 16)
plt.xlabel (r' SINR and SNR, N$_s$ = %d'% n_s, fontsize = 16)

plt.xlim([-20,42])
#plt.ylabel('Probability of Coverage, P', fontsize = 16)
plt.ylabel('CDF ', fontsize = 16)

plt.grid()
if power_control is True:
    plt.savefig('snr_sinr_stream=%d_height=%dm_ptrl.png'%(n_s, height))
else:
    plt.savefig('snr_sinr_stream=%d_height=%dm.png' % (n_s, height))
plt.show()



