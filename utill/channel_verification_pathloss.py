import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xx = np.linspace(-200, 200, 60)
zz = np.linspace(30, 150, 60)
frequency = 28e9

xx_bin, zz_bin = np.meshgrid(xx, zz)
xx_bin, zz_bin = xx_bin.reshape(-1), zz_bin.reshape(-1)
yy_bin = np.zeros_like(zz_bin)#+ np.random.uniform(0,20,size = zz_bin.shape)

rx_dist = np.column_stack((xx_bin, yy_bin, zz_bin))
tx_dist = np.zeros_like(rx_dist)
tx_dist[:,2] = 10

dist_vect = rx_dist - tx_dist
distance = np.linalg.norm(dist_vect, axis = 1)

pl_list = np.array([])
ind_list = np.array([])
for i, h in enumerate(zz_bin):
    #pl_free = 20 *np.log10(distance[i]) + 20*np.log10(frequency) - 147.55
    pl1 = 32.4+(43.2-7.6*np.log10(h))*np.log10(distance[i]) + 20*np.log10(frequency/1e9)
    pl2 = 30.9 + (22.25-0.5*np.log10(h))*np.log10(distance[i]) + 20*np.log10(frequency/1e9)
    pl_list = np.append(pl_list, pl1-pl2)
    if pl2>pl1:
        ind_list = np.append(ind_list, [1])
    else:
        ind_list = np.append(ind_list, [0])
pl_list = pl_list.reshape(60, 60)
ind_list = ind_list.reshape(60,60)
pl_list = np.flip(pl_list,axis=0)
#plt.imshow(pl_list)
#plt.colorbar()
#plt.show()

noise_power = -80
UAV_Height  =60
n_iter = 30
def get_df(dir_="data_with_nn", freq = 28e9):
    df = pd.DataFrame()
    for t in range(n_iter):
        if freq == 28e9 or freq == 73e9 or freq ==140e9:
            data = pd.read_csv('%s/data_%dm_height/uplink_interf_and_data_%dG_%d.txt' % (
                dir_, UAV_Height,int(freq/1e9), t), delimiter='\t')
        else:
            data = pd.read_csv('../data_with_3gpp_channel/data_%dm_height/uplink_interf_and_data_2G_%d.txt' % (
                UAV_Height, t), delimiter='\t')

        df = pd.concat([df,data])
        #intra_itf = df['itf_los']
        #inter_itf = df['itf_nlos']
        l_f = df['l_f']
        #intra_itf_lin = 10 ** (0.1 * intra_itf)
        #inter_itf_lin = 10 ** (0.1 * inter_itf)
        #UE_power = 23 #df['tx_power']
        #noise_and_itf = noise_power + inter_itf_lin + intra_itf_lin
        #SINR = UE_power + l_f - 10 * np.log10(noise_and_itf)
        # SINR = l_f - 10*np.log10(noise_and_itf)

        #SNR = UE_power + l_f - 10 * np.log10(noise_power)
    return df

df = get_df()
df = df[df['ue_type']=='uav']
los_associate = df['link_state']

#itf_los = df['itf_los']
#itf_nlos = df['itf_nlos']

los_count = df['itf_los']
los_count = los_count[los_count!=-200]
nlos_count = df['itf_nlos']
t_count = los_count + nlos_count
print (los_count.sum()/t_count.sum())
#print (np.sum(los_associate==1)/len(df))
plt.plot(np.sort(los_count), np.linspace(0,1,len(los_count)), label = 'LOS')
plt.plot(np.sort(nlos_count), np.linspace(0,1, len(nlos_count)), label = 'NLOS')
#plt.plot(np.sort(t_count), np.linspace(0,1,len(t_count)), label = 'LOS + NLOS')
plt.legend()
plt.grid()
plt.show()
