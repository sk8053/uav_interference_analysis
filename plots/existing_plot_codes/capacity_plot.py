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
def get_df(UAV_Height, dir_=None, orth = False, n =0):
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
    if orth is True:
        capacity = n/10*BW*np.log2(1 + 10**(0.1*SINR))
    else:
        capacity = BW*np.log2(1 + 10**(0.1*SINR))
    capacity = np.array(capacity)
    arg_ind = np.argsort(capacity)
    L = len(capacity)
    ind_5, ind_50, ind_95 = arg_ind[int(L*0.05)], arg_ind[int(L*0.5)], arg_ind[int(L*0.95)]
    ind_arrays = np.array([ind_5, ind_50, ind_95])
    return  capacity, capacity[ind_arrays]

dir_1 = 'data_with_nn_1_tilt_-12_new_ant'
dir_1_o = 'data_with_nn_1_tilt_-12_new_ant_orthogonal'
dir_2 = 'data_with_nn_2_tilt_-12_new_ant'
dir_2_o = 'data_with_nn_2_tilt_-12_new_ant_orthogonal'
dir_3 ='data_with_nn_3_tilt_-12_new_ant'
dir_3_o ='data_with_nn_3_tilt_-12_new_ant_orthogonal'
dir_4 ='data_with_nn_4_tilt_-12_new_ant'
dir_4_o ='data_with_nn_4_tilt_-12_new_ant_orthogonal'

df1_cap, df1_cap_5_50_95 = get_df(60, dir_ = dir_1)
df1_cap_o,df1_cap_5_50_95_o = get_df(60, dir_ = dir_1_o, orth=True, n = 2)

df2_cap, df2_cap_5_50_95 = get_df(60, dir_ = dir_2)
df2_cap_o, df2_cap_5_50_95_o = get_df(60, dir_ = dir_2_o, orth=True, n =4)

df3_cap, df3_cap_5_50_95 = get_df(60, dir_ = dir_3)
df3_cap_o, df3_cap_5_50_95_o = get_df(60, dir_ = dir_3_o, orth=True, n =6)

df4_cap, df4_cap_5_50_95 = get_df(60, dir_ = dir_4)
df4_cap_o, df4_cap_5_50_95_o = get_df(60, dir_ = dir_4_o, orth=True, n =8)

'''
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
plt.legend(fontsize =14)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlabel('Rate (Gbps)', fontsize = 16)
plt.ylabel('CDF', fontsize = 16)
plt.grid()
'''
plt.figure(figsize=(16,4))

#plt.tight_layout(rect=[0,0,1,1])
names = ['5', '50', '95']
c_map = ['r','g','b']
width = 0.3

plt.subplot(1,4,1)
for i, name in enumerate(names):
    if i ==0:
        plt.bar(i-width/2, df1_cap_5_50_95[i],width=width, color = 'r', edgecolor = 'w', label = 'sharing')
        plt.bar(i+width/2, df1_cap_5_50_95_o[i],width=width, color = 'b', hatch = '///',edgecolor = 'w', label = 'no sharing')
    else:
        plt.bar(i - width / 2, df1_cap_5_50_95[i], width=width, color='r', edgecolor='w')
        plt.bar(i + width / 2, df1_cap_5_50_95_o[i], width=width, color='b', hatch='///', edgecolor='w')
ax = plt.gca()
ax.set_xticks([0,1,2])
ax.set_xticklabels(['5','50','95'])
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.grid()
plt.title('1-connection case', fontsize = 16)
#plt.ylim([0, 4.5e9])
plt.ylabel ('Capacity per UE [Gbps]', fontsize = 16)
#plt.yscale('log')
#b, t = plt.ylim()

plt.subplot(1,4,2)
for i, name in enumerate(names):
    if i ==0:
        plt.bar(i-width/2, df2_cap_5_50_95[i],width=width, color = 'r', edgecolor = 'w', label = 'sharing')
        plt.bar(i+width/2, df2_cap_5_50_95_o[i],width=width, color = 'b', hatch = '///',edgecolor = 'w', label = 'no sharing')
    else:
        plt.bar(i - width / 2, df2_cap_5_50_95[i], width=width, color='r', edgecolor='w')
        plt.bar(i + width / 2, df2_cap_5_50_95_o[i], width=width, color='b', hatch='///', edgecolor='w')
ax = plt.gca()
ax.set_xticks([0,1,2])
ax.set_xticklabels(['5','50','95'])
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylim([0, 4.5e9])
plt.grid()
plt.title('2-connection case', fontsize = 16)
#plt.yscale('log')
plt.subplot(1,4,3)
for i, name in enumerate(names):
    if i ==0:
        plt.bar(i-width/2, df3_cap_5_50_95[i],width=width, color = 'r', edgecolor = 'w', label = 'sharing')
        plt.bar(i+width/2, df3_cap_5_50_95_o[i],width=width, color = 'b', hatch = '///',edgecolor = 'w', label = 'no sharing')
    else:
        plt.bar(i - width / 2, df3_cap_5_50_95[i], width=width, color='r', edgecolor='w')
        plt.bar(i + width / 2, df3_cap_5_50_95_o[i], width=width, color='b', hatch='///', edgecolor='w')
ax = plt.gca()
ax.set_xticks([0,1,2])
ax.set_xticklabels(['5','50','95'])
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylim([0, 4.5e9])
plt.grid()
plt.title('3-connection case', fontsize = 16)
#plt.yscale('log')
plt.subplot(1,4,4)
for i, name in enumerate(names):
    if i ==0:
        plt.bar(i-width/2, df4_cap_5_50_95[i],width=width, color = 'r', edgecolor = 'w', label = 'sharing')
        plt.bar(i+width/2, df4_cap_5_50_95_o[i],width=width, color = 'b', hatch = '///',edgecolor = 'w', label = 'no sharing')
    else:
        plt.bar(i - width / 2, df4_cap_5_50_95[i], width=width, color='r', edgecolor='w')
        plt.bar(i + width / 2, df4_cap_5_50_95_o[i], width=width, color='b', hatch='///', edgecolor='w')
ax = plt.gca()
ax.set_xticks([0,1,2])
ax.set_xticklabels(['5','50','95'])
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylim([0, 4.5e9])
plt.grid()
plt.title('4-connection case', fontsize = 16)
plt.suptitle('Perentile', x=0.5, y= 0.05, fontsize=16, fontweight = '550')

#plt.xlabel('Percentile', fontsize = 16)

plt.legend(bbox_to_anchor=(-0.25,-0.08), loc="upper left",fontsize = 16, ncol = 2)

#plt.yscale('log')
plt.show()
#plt.savefig('snr_sinr_mmse_%dG_%dm_height_UAV=gUE=%d_ISD_%d.png'%(int(freq/1e9), UAV_Height,n_UAV,ISD ), dpi = 1200)




