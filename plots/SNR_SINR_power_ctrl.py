import numpy as np

import sys
import pathlib
ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path+'/uav_interference_analysis/src/')
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from power_optimizer import power_optimization
from tqdm import tqdm
import random

random.seed(10)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n',action='store',default=30,type= int,\
    help='number of iteration')
parser.add_argument('--f',action='store',default=28e9,type= float,\
    help='frequency')

plt.rcParams["font.family"] = "Times New Roman"
args = parser.parse_args()
n_iter = args.n
freq = args.f


ISD = 200
#ratio = 50
#n_UAV = 60
n_iter =30
UE_power = 23
KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6

KT_lin = 10**(0.1*KT)
NF_lin = 10**(0.1*NF)
def get_df(dir_=None, n_s=1, opt = False):
    df = pd.DataFrame()
    t_n = np.array([])
    for t in tqdm(np.arange(n_iter)):

        #path_ = '../%s/uplink_interf_and_data_28G_%d.txt' % (dir_,  t)
        path_ = '../%s/uplink_itf_ns=%d_h=60_28G_%d.txt' % (dir_,n_s, t)

        data = pd.read_csv(path_, delimiter='\t')
        #print (data.shape)
        if opt is True:
            power_optimizer = power_optimization(l_f=data['l_f'], ue_ids=data['ue_id'],
                                                 itf_no_ptx_list=data['itf_no_ptx_list'],
                                                 itf_id_list=data['itf_id_list'], n_s = n_s)
            P_tx, SINR_list = power_optimizer.get_optimal_power_and_SINR(minstep=1e-6, debug=False)
            data['tx_power_opt'] = 10 * np.log10(P_tx)
            data['SINR_opt'] = SINR_list
            data.to_csv(path_, sep='\t', index=False)

        df = pd.concat([df,data])

    df = df[df['ue_type']=='g_ue']
    noise_power_dB = 10*np.log10(BW) + KT+NF
    noise_power_lin = KT_lin * NF_lin * BW
    tx_power = df['tx_power']
    l_f = df['l_f']
    intra_itf = df['intra_itf']
    inter_itf = df['inter_itf']

    if n_s ==1:
        intra_itf_lin = 0
    else:
        intra_itf_lin = 10 ** (0.1 * intra_itf)

    inter_itf_lin = 10**(0.1*inter_itf)
    total_itf_lin = intra_itf_lin +  inter_itf_lin
    noise_and_itf_dB = 10*np.log10(noise_power_lin + total_itf_lin)

    SINR = tx_power + l_f - noise_and_itf_dB #df['SINR']
    SNR = tx_power + l_f - noise_power_dB #df['SNR'])
    SINR_opt = np.array([])
    #if opt is True:
    SINR_opt = df['SINR_opt'] #np.array([]) #
    #SNR = df['SNR']

    #Tx_power = df['tx_power']
    #Tx_power_opt = df['tx_power_opt']
    return   np.array(SINR), SNR, SINR_opt#Tx_power,Tx_power_opt,  np.array(SINR), np.array(SINR_opt), SNR

n_s = 4
height = 60
opt = True
#dir_1 = 'new_data_ptrl/%d_stream/%dm_height/UE_8_UAV_0'%(n_s, height)
#dir_1_f = 'new_data/%d_stream/%dm_height/UE_8_UAV_0'%(n_s, height)
#dir_2 = 'new_data_ptrl/%d_stream/%dm_height/UE_7_UAV_1'%(n_s, height)
#dir_3 ='new_data_ptrl/%d_stream/%dm_height/UE_6_UAV_2'%(n_s, height)

#dir_ ='test_data/dedicated_1_1'
dir_f_ ='test_data/1_stream/UE_4_UAV_4'
dir_ ='test_data/1_stream_ptrl/UE_4_UAV_4'

#dir_ = 'new_data_ptrl/%d_stream/%dm_height/UE_4_UAV_4'%(n_s, height)
#dir_f_ = 'new_data/%d_stream/%dm_height/UE_4_UAV_4'%(n_s, height)

df1_sinr_f, df1_snr_f,df1_sinr_opt = get_df(dir_ = dir_f_, n_s = 1, opt=opt)

#df1_sinr,  df1_sn_,_ = get_df(dir_ = dir_, n_s = 2, opt= opt) #, full_tx = True)

#plt.plot(np.sort(df1_sinr), np.linspace(0,1,len(df1_sinr)), label = 'SINR, 3gpp power control', color = 'r')
plt.plot(np.sort(df1_sinr_f), np.linspace(0,1,len(df1_sinr_f)), label = 'SINR, full_tx_power control', color = 'k')
plt.plot(np.sort(df1_sinr_opt), np.linspace(0,1,len(df1_sinr_opt)), label = 'SINR, optimal control', color = 'g')
#plt.plot(np.sort(df1_snr_f), np.linspace(0,1,len(df1_snr_f)), 'b:',label = 'SNR')




plt.legend(fontsize =15)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.xlabel('SINR (dB)', fontsize = 16)
#plt.xlabel (r' SINR and SNR Threshold T (dB), N$_s$ = %d'% n_s, fontsize = 16)
#plt.xlabel('Transmit power (dBm)', fontsize =16)
plt.ylabel('CDF', fontsize = 16)

plt.grid()

plt.show()
#plt.savefig('snr_sinr_mmse_%dG_%dm_height_UAV=gUE=%d_ISD_%d.png'%(int(freq/1e9), UAV_Height,n_UAV,ISD ), dpi = 1200)




