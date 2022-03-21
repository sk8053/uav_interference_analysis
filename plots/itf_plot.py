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
def get_df(UAV_Height, dir_=None, orth = False, n =1):
    noise_power = KT_lin * NF_lin * BW
    df = pd.DataFrame()
    t_n = np.array([])
    SINR_list = []
    for t in range(n_iter):
        if freq == 28e9:
            data = pd.read_csv('../%s/data_%dm_height/uplink_interf_and_data_28G_%d.txt' % (
                dir_, UAV_Height,  t), delimiter='\t')
            #t_n = np.append(t_n, np.repeat([len(data)/2-1], len(data[data['ue_type']=='uav'])))
        else:
            data = pd.read_csv('../data_with_3gpp_channel/data_%dm_height/uplink_interf_and_data_2G_%d.txt' % (
                UAV_Height, t), delimiter='\t')
        bs_id_list = np.unique(data['bs_id'])

        for bs_id in bs_id_list:
            SINRs = data['SINR'][bs_id == data['bs_id']]

            SINR_sum = (10**(0.1*SINRs)).sum()

            SINR_list.append(SINR_sum)
        df = pd.concat([df,data])
    SINR_list = np.array(SINR_list)

    #df = df[df['link_state']==2]
    df_gue = df[df['ue_type'] == 'g_ue']
    df_uav = df[df['ue_type'] == 'uav']

    intra_itf = df['intra_itf']
    inter_itf = df['inter_itf']
    
    intra_itf_lin = 10**(0.1*intra_itf)
    inter_itf_lin = 10**(0.1*inter_itf)

    total_itf =  inter_itf_lin + intra_itf_lin
    total_itf_db = 10*np.log10(total_itf) - 10*np.log10(noise_power)


    #noise_and_itf = noise_power + inter_itf_lin + intra_itf_lin
    #INR  = 10*np.log10(total_itf) - 10*np.log10(noise_power)
    #SINR = df['SINR']
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
    SINR = df['SINR']
    #if orth is True:
    capacity = (n*BW/10)*np.log2(1 + 10**(0.1*SINR))
    #else:
        #capacity = n*BW*np.log2(1 + 10**(0.1*SINR))

    return  np.array(total_itf_db),np.array(capacity)

dir_1 = 'data_with_nn_1_tilt_-12_new_ant'
dir_1_o = 'data_with_nn_1_tilt_-12_new_ant_orthogonal'

dir_2 = 'data_with_nn_2_tilt_-12_new_ant'
dir_2_o = 'data_with_nn_2_tilt_-12_new_ant_orthogonal'

dir_3 = 'data_with_nn_3_tilt_-12_new_ant'
dir_3_o ='data_with_nn_3_tilt_-12_new_ant_orthogonal'

dir_4 = 'data_with_nn_4_tilt_-12_new_ant'
dir_4_o ='data_with_nn_4_tilt_-12_new_ant_orthogonal'

df1_total, df1_cap = get_df(60, dir_ = dir_1, n =2)
df2_total,df2_cap = get_df(60, dir_ = dir_2, n=4)
df3_total,df3_cap = get_df(60, dir_ = dir_3, n= 6)
df4_total,df4_cap = get_df(60, dir_ = dir_4, n=8)

df1_total_o, df1_cap_o = get_df(60, dir_ = dir_1_o, orth=True, n =1)
df2_total_o,df2_cap_o = get_df(60, dir_ = dir_2_o, orth=True, n =1)
df3_total_o,df3_cap_o = get_df(60, dir_ = dir_3_o, orth=True, n =1)
df4_total_o,df4_cap_o = get_df(60, dir_ = dir_4_o, orth=True, n =1)

plt.figure(figsize=(8,6))
'''
plt.subplot(1,2,1)

if True:
    plt.plot(np.sort(df1_total), np.linspace(0,1,len(df1_total)),'b', label = 'sharing, 1-connection', lw = 2)
    plt.plot(np.sort(df2_total), np.linspace(0,1,len(df2_total)),'g', label = 'sharing, 2-connection', lw = 2)
    plt.plot(np.sort(df3_total), np.linspace(0,1,len(df3_total)),'k', label = 'sharing, 3-connection', lw =2)
    plt.plot(np.sort(df4_total), np.linspace(0,1,len(df4_total)),'r', label = 'sharing, 4-connection', lw=2)
if True:
    plt.plot(np.sort(df1_total_o), np.linspace(0,1,len(df1_total_o)),'b-.', label = 'no sharing, 1-connection', lw = 2)
    plt.plot(np.sort(df2_total_o), np.linspace(0,1,len(df2_total_o)),'g-.', label = 'no sharing, 2-connection', lw = 2)
    plt.plot(np.sort(df3_total_o), np.linspace(0,1,len(df3_total_o)),'k-.', label = 'no sharing, 3-connection', lw =2)
    plt.plot(np.sort(df4_total_o), np.linspace(0,1,len(df4_total_o)),'r-.', label = 'no sharing, 4-connection', lw=2)

plt.legend(fontsize =16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.xlabel('INR (dB)', fontsize = 16)

plt.ylabel('CDF', fontsize = 16)

plt.grid()

'''
#plt.subplot(1,2,2)
if True:
    plt.plot(np.sort(df1_cap), np.linspace(0,1,len(df1_cap)),'b', label = 'sharing, 1-connection', lw = 2)
    plt.plot(np.sort(df2_cap), np.linspace(0,1,len(df2_cap)),'g', label = 'sharing, 2-connection', lw = 2)
    plt.plot(np.sort(df3_cap), np.linspace(0,1,len(df3_cap)),'k', label = 'sharing, 3-connection', lw =2)
    plt.plot(np.sort(df4_cap), np.linspace(0,1,len(df4_cap)),'r', label = 'sharing, 4-connection', lw=2)
if True:
    plt.plot(np.sort(df1_cap_o), np.linspace(0,1,len(df1_cap_o)),'b-.', label = 'no sharing, 1-connection', lw = 2)
    plt.plot(np.sort(df2_cap_o), np.linspace(0,1,len(df2_cap_o)),'g-.', label = 'no sharing, 2-connection', lw = 2)
    plt.plot(np.sort(df3_cap_o), np.linspace(0,1,len(df3_cap_o)),'k-.', label = 'no sharing, 3-connection', lw =2)
    plt.plot(np.sort(df4_cap_o), np.linspace(0,1,len(df4_cap_o)),'r-.', label = 'no sharing, 4-connection', lw=2)


plt.tight_layout(rect=[0,0,1,1])
#plt.legend(bbox_to_anchor=(0.23,0.5), loc="upper left", fontsize =14)
plt.legend(fontsize = 14, loc = 'lower center')
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.xlabel('Rate (Gbps)', fontsize = 16)

plt.ylabel('CDF', fontsize = 16)
plt.grid()

p = 0.4

X_detail1, y_detail1 = np.sort(df1_cap)[:int(p*len(df1_cap))], np.linspace(0,1,len(df1_cap))[:int(p*len(df1_cap))]
X_detail2, y_detail2 = np.sort(df2_cap)[:int(p*len(df2_cap))], np.linspace(0,1,len(df2_cap))[:int(p*len(df2_cap))]
X_detail3, y_detail3 = np.sort(df3_cap)[:int(p*len(df3_cap))], np.linspace(0,1,len(df3_cap))[:int(p*len(df3_cap))]
X_detail4, y_detail4 = np.sort(df4_cap)[:int(p*len(df4_cap))], np.linspace(0,1,len(df4_cap))[:int(p*len(df4_cap))]

X_detail1_o, y_detail1_o = np.sort(df1_cap_o)[:int(p*len(df1_cap_o))], np.linspace(0,1,len(df1_cap_o))[:int(p*len(df1_cap_o))]
X_detail2_o, y_detail2_o = np.sort(df2_cap_o)[:int(p*len(df2_cap_o))], np.linspace(0,1,len(df2_cap_o))[:int(p*len(df2_cap_o))]
X_detail3_o, y_detail3_o = np.sort(df3_cap_o)[:int(p*len(df3_cap_o))], np.linspace(0,1,len(df3_cap_o))[:int(p*len(df3_cap_o))]
X_detail4_o, y_detail4_o = np.sort(df4_cap_o)[:int(p*len(df4_cap_o))], np.linspace(0,1,len(df4_cap_o))[:int(p*len(df4_cap_o))]

sub_axes = plt.axes([.62, .55, .35, .33])
sub_axes.plot(X_detail1, y_detail1, c='b')
sub_axes.plot(X_detail2, y_detail2, c='g')
sub_axes.plot(X_detail3, y_detail3, c='k')
sub_axes.plot(X_detail4, y_detail4, c='r')

sub_axes.plot(X_detail1_o, y_detail1_o, 'b-.')
sub_axes.plot(X_detail2_o, y_detail2_o, 'g-.')
sub_axes.plot(X_detail3_o, y_detail3_o, 'k-.')
sub_axes.plot(X_detail4_o, y_detail4_o, 'r-.')
plt.grid()

plt.show()
#plt.savefig('snr_sinr_mmse_%dG_%dm_height_UAV=gUE=%d_ISD_%d.png'%(int(freq/1e9), UAV_Height,n_UAV,ISD ), dpi = 1200)




