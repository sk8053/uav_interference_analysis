import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_s',action='store',default=2,type= int,\
    help='number of iteration')
parser.add_argument('--h',action='store',default=60,type= int,\
    help='frequency')

plt.rcParams["font.family"] = "Times New Roman"
args = parser.parse_args()
n_s = args.n_s
height = args.h

n_iter = 30
ISD = 200
#ratio = 50
#n_UAV = 60
freq = 28e9

UE_power = 23
KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6

KT_lin = 10**(0.1*KT)
NF_lin = 10**(0.1*NF)
def get_df(UAV_Height, dir_=None, orth = False, n =0):
    df = pd.DataFrame()
    t_n = np.array([])
    for t in range(n_iter):
        if freq == 28e9:
            data = pd.read_csv('../%s/uplink_interf_and_data_28G_%d.txt' % (
                dir_,  t), delimiter='\t')
            #t_n = np.append(t_n, np.repeat([len(data)/2-1], len(data[data['ue_type']=='uav'])))
        else:
            data = pd.read_csv('../data_with_3gpp_channel/data_%dm_height/uplink_interf_and_data_2G_%d.txt' % (
                UAV_Height, t), delimiter='\t')
        df = pd.concat([df,data])

    #df = df[df['link_state']==2]
    df_gue = df[df['ue_type'] == 'g_ue']

    intra_itf = df_gue['intra_itf']

    if n ==1:
        intra_itf_lin = 0
    else:
        intra_itf_lin = 10 ** (0.1 * intra_itf)

    inter_itf = df_gue['inter_itf']

    inter_itf_lin = 10**(0.1*inter_itf)

    total_itf_gue =  inter_itf_lin + intra_itf_lin

    df_uav = df[df['ue_type'] == 'uav']
    intra_itf = df_uav['intra_itf']
    inter_itf = df_uav['inter_itf']

    if n ==1:
        intra_itf_lin = 0
    else:
        intra_itf_lin = 10 ** (0.1 * intra_itf)

    inter_itf_lin = 10 ** (0.1 * inter_itf)

    total_itf_uav = inter_itf_lin + intra_itf_lin

    #total_itf_db = 10*np.log10(total_itf)
    noise_power = KT_lin*NF_lin*BW

    noise_and_itf_gue = noise_power + total_itf_gue
    noise_and_itf_uav = noise_power + total_itf_uav

    #INR  = 10*np.log10(total_itf) - 10*np.log10(noise_power)

    g_ue_tx_power = df_gue['tx_power']
    uav_tx_power = df_uav['tx_power']
    SINR_gue = g_ue_tx_power + df_gue['l_f'] - 10*np.log10(noise_and_itf_gue)
    SINR_uav = uav_tx_power + df_uav['l_f'] - 10*np.log10(noise_and_itf_uav)

    capacity_gue = n/10*BW*np.log2(1 + 10**(0.1*SINR_gue))
    capacity_uav = n / 10 * BW * np.log2(1 + 10 ** (0.1 * SINR_uav))

    capacity_gue = np.array(capacity_gue)
    capacity_uav = np.array(capacity_uav)

    #arg_ind = np.argsort(capacity)
    #L = len(capacity)
    #ind_5, ind_50, ind_95 = arg_ind[int(L*0.05)], arg_ind[int(L*0.5)], arg_ind[int(L*0.95)]
    #ind_arrays = np.array([ind_5, ind_50, ind_95])
    return  capacity_gue, capacity_uav

power_control = False
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

capacity80_gue, capacity80_uav  = get_df(height, dir_ = dir_1, n = n_s)
capacity71_gue,capacity71_uav  = get_df(height, dir_ = dir_2, n = n_s)
capacity62_gue, capacity62_uav = get_df(height, dir_ = dir_3, n= n_s)
capacity53_gue, capacity53_uav = get_df(height, dir_ = dir_4, n = n_s)
capacity44_gue, capacity44_uav  = get_df(height, dir_ = dir_5, n= n_s)


plt.figure(figsize=(9,7))

plt.subplot(1,2,1)
plt.plot(np.sort(capacity80_gue), np.linspace(0,1,len(capacity80_gue)), 'r', label = 'UAV = 0 %')
plt.plot(np.sort(capacity71_gue), np.linspace(0,1,len(capacity71_gue)), 'g', label = 'UAV = 12.5 %')
plt.plot(np.sort(capacity62_gue), np.linspace(0,1,len(capacity62_gue)), 'b', label = 'UAV = 25 %')
plt.plot(np.sort(capacity53_gue), np.linspace(0,1,len(capacity53_gue)), 'k', label = 'UAV = 37.5 %')
plt.plot(np.sort(capacity44_gue), np.linspace(0,1,len(capacity44_gue)), 'c', label = 'UAV = 50 %')
plt.grid()
plt.xticks(fontsize =15)
plt.yticks(fontsize =15)
plt.title('Capacity of Ground UEs', fontsize = 15)
plt.xlim([0, 1.8e9])
plt.ylabel ('CDF', fontsize = 15)

plt.subplot(1,2,2)
plt.plot(np.sort(capacity80_uav), np.linspace(0,1,len(capacity80_uav)), 'r', label = 'UAV = 0 %')
plt.plot(np.sort(capacity71_uav), np.linspace(0,1,len(capacity71_uav)), 'g', label = 'UAV = 12.5 %')
plt.plot(np.sort(capacity62_uav), np.linspace(0,1,len(capacity62_uav)), 'b', label = 'UAV = 25 %')
plt.plot(np.sort(capacity53_uav), np.linspace(0,1,len(capacity53_uav)), 'k', label = 'UAV = 37.5 %')
plt.plot(np.sort(capacity44_uav), np.linspace(0,1,len(capacity44_uav)), 'c', label = 'UAV = 50 %')
plt.title('Capacity of UAVs', fontsize = 15)
#plt.legend(fontsize =15)
plt.legend(bbox_to_anchor=(.48,0.26), loc="upper left",fontsize = 13, ncol = 1)
plt.xticks(fontsize =15)
plt.yticks(fontsize =15)
plt.xlim([0, 1.8e9])
plt.suptitle(r'Capacity (N$_s$ = %d) '%n_s, x=0.5, y= 0.03, fontsize=16, fontweight = '550')
plt.grid()
#plt.tight_layout(rect = (0,0,1,1), pad = 0.7)
#
if power_control is False:
    plt.savefig('capacity_ns_%d_height_%d.png'%(n_s, height))
else:
    plt.savefig('capacity_ns_%d_height_%d_ptrl.png' % (n_s, height))
#plt.show()

