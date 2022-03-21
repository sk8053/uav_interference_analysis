import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pathlib

ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path + '/src/')
data = dict()

feature = 'interference'
#feature = 'SINR'
#feature = "UE_TX_Power"
#alpha = '0.5'
#alphas = np.repeat([alpha],6)
#alphas = ['0.1','0.3','0.5', '0.8','1.0']#,'0.5','0.8','1.0']#,'0.8','1.0']
#alphas = ['0.1','0.3','0.5', '0.8', '1.0']#, '0.8','1.0']
'''
Ps = ['40','60','80'] #,'80']#,'60','70','90','100']
l_t = ['-','-.',':', '--',' ']
colors = ['b','g','r','c','k']
for i, P0 in enumerate(Ps):
    for j, alpha in enumerate(alphas):
        dir = '../data/ptr_%s/' % P0
        file_name = 'uplink_interf_and_data_%s_%s_.txt'%(P0,alpha)
        data[alpha] = pd.read_csv(dir + file_name, delimiter = '\t', index_col = False)
        SINR = data[alpha][feature]
        plt.plot(np.sort(SINR), np.linspace(0,1,len(SINR)),colors[j]+l_t[i], label = '$\\alpha$ = %s, $P_0$ = -%s'%(alpha, P0))
'''
file_name = '../data/uplink_interf_and_data_30_.txt'
data_30 = pd.read_csv(file_name, delimiter = '\t', index_col = False)
file_name = '../data/uplink_interf_and_data_60_.txt'
data_60 = pd.read_csv(file_name, delimiter = '\t', index_col = False)
file_name = '../data/uplink_interf_and_data_120_.txt'
data_120 = pd.read_csv(file_name, delimiter = '\t', index_col = False)

def plot_snir_snr(data, height, l='-', enable_SNR = False):
    SINR_gue = data[data['UE_type']=='g_ue']['SINR']-6
    SNR_gue = data[data['UE_type']=='g_ue']['SNR']-6
    SINR_uav = data[data['UE_type']=='uav']['SINR']-6
    SNR_uav = data[data['UE_type']=='uav']['SNR']-6
    SNR, SINR = data['SNR']-6, data['SINR']-6

    #print(len(SNR_gue), len(SNR_uav))
    #SNR_60, SINR_60 = data_60['SNR']-6, data_60['SINR']-6

    #plt.plot(np.sort(SINR), np.linspace(0, 1, len(SINR)), 'r' + '-', label='SINR (UAV + gUE) at %sm height' % height)
    #plt.plot(np.sort(SINR_uav), np.linspace(0,1,len(SINR_uav)),'r'+'-.', label = 'SINR by UAV at %sm height'%height)
    #plt.plot(np.sort(SINR_gue), np.linspace(0, 1, len(SINR_gue)), color='tab:red', linestyle=':',label='SINR by gUE at %sm height' % height)
    if enable_SNR is False:
        #plt.plot(np.sort(SINR_gue), np.linspace(0, 1, len(SINR_gue)), color='tab:red', linestyle=l,
        #     label='SINR by gUE at %sm height' % height)
        #plt.plot(np.sort(SINR_uav), np.linspace(0, 1, len(SINR_uav)), 'r' + l,label='SINR by UAV at %sm height' % height)
        plt.plot(np.sort(SINR), np.linspace(0, 1, len(SINR)), 'r' + l, label='SINR (UAV + gUE) at %sm height' % height)


    #plt.plot(np.sort(SNR), np.linspace(0, 1, len(SNR)), 'g' + '-', label='SNR (UAV + gUE) at %sm height' % height)
    #plt.plot(np.sort(SNR_uav), np.linspace(0,1,len(SNR_uav)),'g'+'-.', label = 'SNR by UAV at %sm height'%height)
    #plt.plot(np.sort(SNR_gue), np.linspace(0, 1, len(SNR_gue)), 'g' + ':', label='SNR by gUE at %sm height' % height)

    #print (len(SINR_uav), len(SINR_gue))
    else:
        #plt.plot(np.sort(SNR_gue), np.linspace(0,1,len(SNR_gue)),color = 'tab:green',linestyle = l,
         #        label = 'SNR by gUE at %sm height'%height)
        #plt.plot(np.sort(SNR_uav), np.linspace(0, 1, len(SNR_uav)), 'g' + l,label='SNR by UAV at %sm height' % height)
        plt.plot(np.sort(SNR), np.linspace(0, 1, len(SNR)), 'g' + l, label='SNR (UAV + gUE) at %sm height' % height)

uav_height = 30
plot_snir_snr(data_30, 30)
plot_snir_snr(data_60,60, l= '-.')
plot_snir_snr(data_120,120, l=':')

plot_snir_snr(data_30, 30, enable_SNR=True)
plot_snir_snr(data_60,60, l= '-.', enable_SNR=True)
plot_snir_snr(data_120,120, l=':', enable_SNR=True)

#plt.plot(np.sort(SINR_60), np.linspace(0,1,len(SINR_60)),'r-.', label = 'SINR (UAV + gUE) at 60m')
#plt.plot(np.sort(SNR_60), np.linspace(0,1,len(SNR_60)),'b-.', label = 'SNR (UAV + gUE) at 60m')

plt.title ('SNR and SINR (dB) ' ) #at %sm'%str(uav_height))
plt.grid()
plt.legend()
#file_name = 'SNR and SINR at %s height'%str(uav_height)
#plt.savefig('/home/sk8053/Downloads/'+file_name)

plt.figure (3)

itf_30 = data_30['interference']
itf_60 = data_60['interference']
itf_120 = data_120['interference']

KT = -174
NF = 6
BW = 200e6
noise = KT + NF + 10 * np.log10(BW)

INR_60 = itf_60 - noise
INR_120 = itf_120 - noise
INR_30 = itf_30 - noise

plt.plot (np.sort(INR_120), np.linspace(0,1,len(INR_120)), 'r', label = "INR (dB) at 120m")
plt.plot (np.sort(INR_60), np.linspace(0,1,len(INR_60)), 'g', label = "INR (dB) at 60m")
plt.plot (np.sort(INR_30), np.linspace(0,1,len(INR_30)), 'k', label = "INR (dB) at 30m")

plt.title ('Interference to noise ratio (dB)')
#plt.title ('Interference Power (dBm)')

plt.grid()
plt.legend()
#file_name = '%s_alphas_%s_%s'%(feature, alphas[0], alphas[1]) +'.png'
#file_name = 'P0_%s_%s'%(feature, Ps[0])+'.png'

plt.show()
