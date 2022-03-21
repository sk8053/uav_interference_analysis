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
def get_df(dir_=None, n = 1, f= 28, uplink = True):
    df = pd.DataFrame()
    t_n = np.array([])
    for t in range(n_iter):
        if uplink is True:
            data = pd.read_csv('../%s/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % (
                    dir_,n,height, f, t), delimiter='\t')
        else:
            data = pd.read_csv('../%s/downlink_itf_ns=%d_h=%d_%dG_%d.txt' % (
                dir_, n, height, f, t), delimiter='\t')
        df = pd.concat([df,data])

    #df_uav = df[df['ue_type'] == 'uav']

    #df = df[df['ue_type']=='g_ue']

    noise_power_dB = 10*np.log10(BW) + KT+NF
    noise_power_lin = KT_lin * NF_lin * BW

    if uplink is True:
        tx_power = df['tx_power']
        l_f = df['l_f']
        intra_itf = df['intra_itf']
        inter_itf = df['inter_itf']

        if n == 1:
            intra_itf_lin = 0
        else:
            intra_itf_lin = 10 ** (0.1 * intra_itf)

        inter_itf_lin = 10 ** (0.1 * inter_itf)
        total_itf_lin = intra_itf_lin + inter_itf_lin
        noise_and_itf_dB = 10 * np.log10(noise_power_lin + total_itf_lin)

        SINR = tx_power + l_f - noise_and_itf_dB
        SNR = tx_power + l_f - noise_power_dB
    else:
        tx_power = 30
        l_f = df['l_f']
        intra_itf = df['intra_itf']
        inter_itf = df['inter_itf']
        itf_ue_uav = df['itf_ue_uav']
        if n == 1:
            intra_itf_lin = 0
        else:
            intra_itf_lin = 10 ** (0.1 * intra_itf)

        inter_itf_lin = 10 ** (0.1 * inter_itf)
        itf_ue_uav_lin = 10**(0.1*itf_ue_uav)
        total_itf_lin = intra_itf_lin + inter_itf_lin + itf_ue_uav_lin
        noise_and_itf_dB = 10 * np.log10(noise_power_lin + total_itf_lin)

        SINR = tx_power + l_f - noise_and_itf_dB
        SNR = tx_power + l_f - noise_power_dB


    p = 0.5
    med_ind = int(len(SINR)*p)
    median_SINR = np.sort(SINR)[med_ind]
    median_SNR = np.sort(SNR)[med_ind]
    print (dir_[-10:], median_SINR, median_SNR)
    INR = 10*np.log10(total_itf_lin) - noise_power_dB
    return  np.array(SINR), SNR, INR #df['itf_ue_uav']-noise_power_dB


dir_1 = 'test_data/2_stream_ptrl/UAV_50'
#dir_2 = 'test_data_dl/2_stream_ptrl/UAV_50'

dir_2 = 'test_data/dedicated_1_1_ptrl/'



df1_sinr, df1_snr, df1_inr = get_df(dir_ = dir_1, n = 2)
df2_sinr,df2_snr, df2_inr= get_df(dir_ = dir_2, n= 1 , uplink= False)

fig, ax = plt.subplots(figsize = (8,6))
lines =[]
if True:
    lines += ax.plot(np.sort(df1_inr), np.linspace(0,1,len(df1_sinr)),'b', label = 'uplink', lw = 2)
    lines += ax.plot(np.sort(df2_inr), np.linspace(0,1,len(df2_sinr)),'g', label = 'downlink', lw = 2)


if False:
    #lines+=  ax.plot(np.sort(df1_snr), np.linspace(0,1,len(df1_snr)),'b:', label = 'UAV = 0 %', lw = 2)
    lines += ax.plot(np.sort(df2_snr), np.linspace(0,1,len(df2_snr)),'g:', label = 'UAV = 12.5 %', lw = 2)
    #lines += ax.plot(np.sort(df3_snr), np.linspace(0,1,len(df3_snr)),'k:', label = 'UAV = 25 %', lw =2)
    lines += ax.plot(np.sort(df4_snr), np.linspace(0,1,len(df4_snr)),'r:', label = 'UAV = 37.5 %', lw=2)
    lines += ax.plot(np.sort(df5_snr), np.linspace(0, 1, len(df5_snr)), 'c:', label='UAV = 50 %', lw=2)



legend1 = ax.legend(fontsize =13)

plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

#plt.xlabel('SINR and SNR (dB)', fontsize = 16)
#plt.xlabel (r' SINR and SNR Threshold T (dB), N$_s$ = %d'% n_s, fontsize = 16)
plt.xlabel (r' SINR and SNR, N$_s$ = %d'% n_s, fontsize = 16)

#plt.xlim([-20,42])
#plt.ylabel('Probability of Coverage, P', fontsize = 16)
plt.ylabel('CDF ', fontsize = 16)

plt.grid()

plt.show()


'''
legend2 = Legend(ax, [lines[2], lines[5]],['SINR', 'SNR'], fontsize = 13,
                 bbox_to_anchor=(-0.02,1.115), loc="upper left", ncol = 2, frameon= False)
                 ax.add_artist(legend2)

'''

