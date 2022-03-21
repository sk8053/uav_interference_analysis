import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
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

    #df = df[df['link_state']==2]
    df_gue = df[df['ue_type'] == 'g_ue']
    df_uav = df[df['ue_type'] == 'uav']

    #ue_x, ue_y, ue_z = df['ue_x'], df['ue_y'], df['ue_z']
    #bs_x, bs_y, bs_z = df['bs_x'], df['bs_y'], df['bs_z']
    #dist_v = np.column_stack((bs_x - ue_x,  bs_y -ue_y))
    #distance = np.array(df_gue['avg_dist_itf'])#np.linalg.norm(dist_v, axis =1)#df['avg_dist'] #np.linalg.norm(dist_v, axis =1)
    avg_dist = []
    #print (distance)
    #for i in np.arange(len(df_gue)):
    #    dist = distance[i][1:-1].split(',')
    #    dist = np.array(dist, dtype = float)
    #    avg_dist.append(np.std(dist))

    #l_f = df['l_f']
    intra_itf = df_gue['intra_itf']
    inter_itf = df_gue['inter_itf']
    intra_itf_lin = 10**(0.1*intra_itf)
    inter_itf_lin = 10**(0.1*inter_itf)

    total_itf =  inter_itf_lin + intra_itf_lin
    total_itf_db = 10*np.log10(total_itf)
    noise_power = KT_lin*NF_lin*BW

    noise_and_itf = noise_power + inter_itf_lin + intra_itf_lin
    INR  = 10*np.log10(total_itf) - 10*np.log10(noise_power)
    SINR = df['SINR']
    INR_intra, INR_inter = intra_itf  - 10*np.log10(noise_power) , inter_itf  - 10*np.log10(noise_power)

    SNR = df['SNR']


    n_los = df_gue['n_los']  # (df['n_los'] + df['n_nlos'])#df['itf_UAV'] - 10*np.log10(noise_power)

    n_nlos = df_gue['n_nlos']
    los_prob = n_los / (n_nlos + n_los)
    INR_uav = df_gue['itf_UAV'] - 10 * np.log10(noise_power)
    los_itf = df_gue['los_itf']

    nlos_itf = df_gue['nlos_itf'] - 10*np.log10(noise_power)
    I = np.where(los_itf > -196)[0]
    los_itf = np.array(los_itf)[I] - 10*np.log10(noise_power)
    return  np.array(SINR), SNR

dir_1 = 'data_with_nn_1_tilt_-12_new_ant'
dir_2 = 'data_with_nn_2_tilt_-12_new_ant'
dir_3 ='data_with_nn_3_tilt_-12_new_ant'
dir_4 ='data_with_nn_4_tilt_-12_new_ant'

df1_sinr, df1_snr = get_df(60, dir_ = dir_1)
df2_sinr,df2_snr= get_df(60, dir_ = dir_2)
df3_sinr,df3_snr= get_df(60, dir_ = dir_3)
df4_sinr,df4_snr= get_df(60, dir_ = dir_4)

if True:
    plt.plot(np.sort(df1_sinr), np.linspace(0,1,len(df1_sinr)),'b', label = 'SINR, 1-connection', lw = 2)
    plt.plot(np.sort(df2_sinr), np.linspace(0,1,len(df2_sinr)),'g', label = 'SINR, 2-connection', lw = 2)
    plt.plot(np.sort(df3_sinr), np.linspace(0,1,len(df3_sinr)),'k', label = 'SINR, 3-connection', lw =2)
    plt.plot(np.sort(df4_sinr), np.linspace(0,1,len(df4_sinr)),'r', label = 'SINR, 4-connection', lw=2)
if True:
    plt.plot(np.sort(df1_snr), np.linspace(0,1,len(df1_snr)),'b-.', label = 'SNR, 1-connection', lw = 2)
    plt.plot(np.sort(df2_snr), np.linspace(0,1,len(df2_snr)),'g-.', label = 'SNR, 2-connection', lw = 2)
    plt.plot(np.sort(df3_snr), np.linspace(0,1,len(df3_snr)),'k-.', label = 'SNR, 3-connection', lw =2)
    plt.plot(np.sort(df4_snr), np.linspace(0,1,len(df4_snr)),'r-.', label = 'SNR, 4-connection', lw=2)



plt.legend(fontsize =13)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.xlabel('INR (dB)', fontsize = 16)

plt.ylabel('CDF', fontsize = 16)

plt.grid()


plt.show()
#plt.savefig('snr_sinr_mmse_%dG_%dm_height_UAV=gUE=%d_ISD_%d.png'%(int(freq/1e9), UAV_Height,n_UAV,ISD ), dpi = 1200)




