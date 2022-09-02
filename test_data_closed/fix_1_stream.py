import pandas as pd
import numpy as np

freq = 28e9
n_iter = 30
KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6

KT_lin = 10**(0.1*KT)
NF_lin = 10**(0.1*NF)

df = pd.DataFrame()

height = 60
n_ue, n_uav = 4, 4
for t in range(n_iter):
    data = pd.read_csv('1_stream/%dm_height/UE_%d_UAV_%d/uplink_interf_and_data_28G_%d.txt' % (
        height,  n_ue, n_uav,t), delimiter='\t')

    tx_power = data['tx_power']
    inter_itf_lin  =  10**(0.1*data['inter_itf'])

    noise_lin = KT_lin*NF_lin*BW
    noise_itf = noise_lin + inter_itf_lin
    SINR = tx_power + data['l_f'] - 10*np.log10(noise_itf)
    data['SINR'] = SINR
    print (SINR)
    data.to_csv('1_stream_new/%dm_height/UE_%d_UAV_%d/uplink_interf_and_data_28G_%d.txt'%(height,n_ue, n_uav ,t), index = False)

