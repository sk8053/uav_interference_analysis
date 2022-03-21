import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

#file_names = ['uplink_interf_and_data_60_1.0.txt','uplink_interf_and_data_60_0.1.txt','uplink_interf_and_data_60_0.25.txt']
def get_sinr(dir,n=30 , height = 60, feature = 'SNR'):
    sinr_list =np.array([])
    snr_list = np.array([])
    itf_list = np.array([])

    for i in range(n):
        file = dir+'uplink_interf_and_data_%d_%d.txt'%(height,i)
        d = pd.read_csv(file, delimiter='\t', index_col = False)
        if feature == 'INR':
            sinr = 10*np.log10(d['itf_gUE'])
            snr = 10*np.log10(d['itf_UAV'])
        else:
            sinr = d['SINR']#[d['ue_type']==0] # 10*np.log10(d['itf_gUE'])#[d['ue_type']==0]
            snr = d['SNR']#[d['ue_type']==0] #10*np.log10(d['itf_UAV'])#[d['ue_type']==0]
        itf = d['interference']#[d['ue_type'] == 0]
        snr_list = np.append(snr_list, snr)
        sinr_list = np.append(sinr_list,sinr)
        itf_list = np.append(itf_list, itf)
    return sinr_list, snr_list,itf_list
h = 60
feature = 'INR'
city_list = ['lon_tok', 'beijing', 'moscow']
color_list = ['r','b', 'k']
city = 'lon_tok'
#for city,c in zip(city_list, color_list):
dir_ = '../data_%d_%s'%(h, city)
dir_50 = dir_ + '/data_50/'
dir_25 = dir_ +'/data_25/'
dir_10 = dir_ +'/data_10/'
dir_5 = dir_ +'/data_5/'

sinr_list_50, snr_list_50, itf = get_sinr(dir_50,30, height = h, feature = feature)
sinr_list_25, snr_list_25,_ = get_sinr(dir_25,30, height = h, feature = feature)
sinr_list_10, snr_list_10,_ = get_sinr(dir_10,30, height = h, feature = feature)
sinr_list_5, snr_list_5,_ = get_sinr(dir_5,30, height = h, feature = feature)
noise = -82
if feature == 'INR':
    plt.plot(np.sort(sinr_list_50-noise), np.linspace(0,1,len(sinr_list_50)),'r--',
             label = '%s, %s by gUE '%(city, feature))
    plt.plot (np.sort(snr_list_50-noise), np.linspace(0,1,len(snr_list_50)),'r',
              label = '%s, %s by UAVs '%(city, feature))
    plt.plot (np.sort(itf-noise), np.linspace(0,1,len(itf)),'k',
              label = '%s, total %s '%(city,feature))
    #plt.xlim(-60,29)

if feature == 'SINR':
    plt.plot(np.sort(sinr_list_50), np.linspace(0,1,len(sinr_list_50)),'r--',  label = 'SINR, 50:50')
    plt.plot(np.sort(sinr_list_25), np.linspace(0,1,len(sinr_list_25)),'g--',  label = 'SINR, 25:75')
    plt.plot(np.sort(sinr_list_10), np.linspace(0,1,len(sinr_list_10)),'b--',  label = 'SINR, 10:90')
    plt.plot(np.sort(sinr_list_5), np.linspace(0,1,len(sinr_list_5)),'k--',  label = 'SINR, 5:95')

elif feature == 'SNR':
    plt.plot (np.sort(snr_list_50), np.linspace(0,1,len(snr_list_50)),'r',  label = 'SNR, 50:50')
    plt.plot (np.sort(snr_list_25), np.linspace(0,1,len(snr_list_25)),'g',  label = 'SNR, 25:75')
    plt.plot (np.sort(snr_list_10), np.linspace(0,1,len(snr_list_10)),'b',  label = 'SNR, 10:90')
    plt.plot (np.sort(snr_list_5), np.linspace(0,1,len(snr_list_5)),'k',  label = 'SNR, 5:95')

plt.title ('%s, height = %dm'%(feature,h))
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xlabel('%s (dB)'%feature, fontsize = 11)
plt.ylabel ('CDF', fontsize =11)
#plt.xlim (-20, 45)
plt.legend(fontsize = 11)
plt.grid()
plt.show()