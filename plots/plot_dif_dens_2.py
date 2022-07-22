import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
BW = 400e6
KT = -174
NF = 6

noise_power = 10*np.log10(BW) + KT + NF

def get_sinr(dir,n=30 , height = 60, feature = 'SNR'):
    sinr_list =np.array([])
    snr_list = np.array([])
    itf_list = np.array([])
    itf_list__ = np.array([])
    dist_list = np.array([])
    for i in range(n):
        file = dir+'uplink_interf_and_data_%d_%d.txt'%(height,i)
        d = pd.read_csv(file, delimiter='\t', index_col = False)
        dist_x, dist_y, dist_z = d['serv_ue_x'] - d['bs_x'],d['serv_ue_y'] - d['bs_y'], d['serv_ue_z'] - d['bs_z']
        distance = np.linalg.norm(np.column_stack((dist_x, dist_y)), axis = 1)
        itf_gUE, itf_UAV, itf_total = np.array([0]), np.array([0]), np.array([0])
        data = dict()
        data = {'itf_gUE':np.array([]), 'itf_UAV':np.array([]), 'itf_total':np.array([])
                , 'SINR':np.array([]), 'SNR':np.array([])}
        if feature == 'INR':
            data['itf_gUE'] = np.append(data['itf_gUE'], 10*np.log10(d['itf_gUE']) - noise_power)
            data['itf_UAV'] = np.append(data['itf_UAV'], 10*np.log10(d['itf_UAV']) - noise_power)
            data['itf_total'] = np.append(data['itf_total'], 10*np.log10(d['itf_gUE']+d['itf_UAV'])-noise_power)
        else:
            data['SINR'] = np.append(data['SINR'], d['SINR'])
            data['SNR'] = np.append(data['SNR'], d['SNR'])

    return data
h =30
feature = 'INR'
city_list = ['lon_tok', 'beijing', 'moscow']
color_list = ['r','b', 'k']
city = 'lon_tok'

dir_200 = '../data_%d_%s_ISD200'%(h, city)

dir_28G = dir_200 + '/data_50/'
dir_2G = dir_200 + '/data_50_2G/'

data_28 = get_sinr(dir_28G,100, height = h, feature = feature)
data_2 = get_sinr(dir_2G,100, height = h, feature = feature)
s = 'total'
if s == 'g_UE':
    data_28 = data_28['itf_gUE']
    data_2 = data_2['itf_gUE']
    f = 'INR of ground UEs'
elif s== 'UAV':
    data_28 = data_28['itf_UAV']
    data_2 = data_2['itf_UAV']
    f = 'INR of UAVs'

elif s == 'total':
    data_28 = data_28['itf_total']
    data_2 = data_2['itf_total']
    f = 'Total INR'
plt.plot(np.sort(data_28), np.linspace(0,1,len(data_28)),'r', label = 'frequency = 28GHz')
plt.plot(np.sort(data_2), np.linspace(0,1,len(data_2)),'k', label = 'frequency = 2GHz')


plt.title ('%s, height = %dm, ISD = %dm'%(f,h, 200), fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel('%s (dB)'%f, fontsize = 13)
plt.ylabel ('CDF', fontsize =13)
plt.tight_layout()
plt.legend(fontsize = 12)
plt.grid()
plt.savefig ('%s_lon_tok_ISD_%d_height%d'%(f,200, h), dpi = 1200)
plt.show()