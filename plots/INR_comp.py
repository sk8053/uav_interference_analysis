import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n',action='store',default=30,type= int,\
    help='number of iteration')
parser.add_argument('--f',action='store',default=28e9,type= float,\
    help='frequency')
parser.add_argument('--n_s',action='store',default=1,type= int,\
    help='number of streams')
parser.add_argument('--h',action='store',default=60,type= int,\
    help='UAV height')

plt.rcParams["font.family"] = "Times New Roman"
args = parser.parse_args()
n_iter = args.n
freq = args.f
height = args.h
n_s= args.n_s

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
power_control = False
def get_df(UAV_Height=60, dir_=None, n=1):
    df = pd.DataFrame()
    t_n = np.array([])
    for t in range(n_iter):
        #if n == 1 and power_control == False:
        #     data = pd.read_csv('../%s/uplink_interf_and_data_28G_%d.txt' % (
        #        dir_,  t), delimiter=',')
        #else:
        data = pd.read_csv('../%s/uplink_interf_and_data_28G_%d.txt' % (
                dir_, t), delimiter='\t')
        #print (data.shape)
        df = pd.concat([df,data])
    df = df[df['ue_type'] == 'g_ue']
    noise_power_dB = 10*np.log10(BW) + KT+NF
    noise_power_lin = KT_lin * NF_lin * BW
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
    total_itf = 10*np.log10(total_itf_lin) - noise_power_dB
    #df = df[df['ue_type'] == 'g_ue']

    noise_power = 10*np.log10(BW) + KT+NF
    SINR =df['SINR'] #df['itf_gUE'] - noise_power

    SNR = df['SNR']#df['itf_UAV'] - noise_power


    return total_itf,  total_itf #, np.array(SINR) #, SNR  df['tx_power'],

if power_control is False:
    dir_1 = 'new_data/%d_stream/%dm_height/UE_8_UAV_0'%(n_s, height)
    dir_2 = 'new_data/%d_stream/%dm_height/UE_7_UAV_1'%(n_s, height)
    dir_3 ='new_data/%d_stream/%dm_height/UE_6_UAV_2'%(n_s, height)
    dir_4 ='new_data/%d_stream/%dm_height/UE_5_UAV_3'%(n_s, height)
    dir_5 = 'new_data/%d_stream/%dm_height/UE_4_UAV_4'%(n_s, height)
else:
    dir_1 = 'new_data_ptrl/%d_stream/%dm_height/UE_8_UAV_0' % (n_s, height)
    dir_2 = 'new_data_ptrl/%d_stream/%dm_height/UE_7_UAV_1' % (n_s, height)
    dir_3 = 'new_data_ptrl/%d_stream/%dm_height/UE_6_UAV_2' % (n_s, height)
    dir_4 = 'new_data_ptrl/%d_stream/%dm_height/UE_5_UAV_3' % (n_s, height)
    dir_5 = 'new_data_ptrl/%d_stream/%dm_height/UE_4_UAV_4' % (n_s, height)

df1_sinr, df1_snr = get_df(dir_ = dir_1, n=n_s)
df2_sinr,df2_snr= get_df( dir_ = dir_2, n=n_s)
df3_sinr,df3_snr= get_df(dir_ = dir_3, n=n_s)
df4_sinr,df4_snr= get_df(dir_ = dir_4, n=n_s)
df5_sinr,df5_snr= get_df(dir_ = dir_5, n=n_s)

fig, ax = plt.subplots()
lines =[]
if True:
    lines += ax.plot(np.sort(df1_sinr), np.linspace(0,1,len(df1_sinr)),'b', label = 'UAV = 0 %', lw = 2)
    lines += ax.plot(np.sort(df2_sinr), np.linspace(0,1,len(df2_sinr)),'g', label = 'UAV = 12.5 %', lw = 2)
    lines += ax.plot(np.sort(df3_sinr), np.linspace(0,1,len(df3_sinr)),'k', label = 'UAV = 25%', lw =2)
    lines += ax.plot(np.sort(df4_sinr), np.linspace(0,1,len(df4_sinr)),'r', label = 'UAV = 37.5 %', lw=2)
    lines += ax.plot(np.sort(df5_sinr), np.linspace(0, 1, len(df5_sinr)), 'c', label='UE = 50 %', lw=2)
if False:
   # plt.plot(np.sort(df1_snr), np.linspace(0,1,len(df1_snr)),'b:', label = 'UE =', lw = 2)
    lines += ax.plot(np.sort(df2_snr), np.linspace(0,1,len(df2_snr)),'g:', label = 'UE = 7, UAV = 1', lw = 2)
    lines += ax.plot(np.sort(df3_snr), np.linspace(0,1,len(df3_snr)),'k:', label = 'UE = 6, UAV = 2', lw =2)
    lines += ax.plot(np.sort(df4_snr), np.linspace(0,1,len(df4_snr)),'r:', label = 'UE = 5, UAV = 3', lw=2)
    lines += ax.plot(np.sort(df5_snr), np.linspace(0, 1, len(df5_snr)), 'c:', label='UE = 4, UAV = 4', lw=2)


legend1 = ax.legend(fontsize =13)
#legend2 = Legend(ax, [lines[2], lines[5]],['Interference by ground UEs', 'Interference by UAVs'], fontsize = 13,
#                 bbox_to_anchor=(-0.02,1.115), loc="upper left", ncol = 2, frameon= False)
#ax.add_artist(legend2)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlim([-35,30])
#plt.xlim(-35,30)
#plt.xlabel('SNR and SINR (dB)', fontsize = 16)
plt.xlabel('INR (dB), '+r'N$_s$ = %d'%n_s, fontsize = 13.5)
#plt.xlabel('UAV Tx Power (dBm)', fontsize = 16)

plt.ylabel('CDF', fontsize = 14)

plt.grid()



if power_control is False:
    plt.savefig('inr_n_s=%d_h=%d.png'%(n_s, height))
else:
    plt.savefig('inr_n_s=%d_h=%d_ptrl.png' % (n_s, height))
#plt.show()
#plt.savefig('snr_sinr_mmse_%dG_%dm_height_UAV=gUE=%d_ISD_%d.png'%(int(freq/1e9), UAV_Height,n_UAV,ISD ), dpi = 1200)




