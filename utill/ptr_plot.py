import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pathlib

ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path + '/src/')
#data = dict()
KT = -174
Noise_figure = 6
#feature = 'interference'
#feature = 'iterference without rxbf'
feature = 'SINR'
#feature2 = 'SNR'
#feature = "UE_TX_Power"

noise =KT+Noise_figure+10*np.log10(400e6)
alphas = ['0.1','0.3','0.5', '0.8','1.0']
#alphas = ['0.8','0.5']
Ps = ['30']#,'80']#,'80','100']
l_t = ['-','-.',':', '--',' ']
colors = ['b','g','r','c','k','y']
for i, P0 in enumerate(Ps):
    for j, alpha in enumerate(alphas):
        dir = '../data/ptr_%s/' % P0
        file_name = 'uplink_interf_and_data_%s_%s_.txt'%(P0,alpha)
        data = pd.read_csv(dir + file_name, delimiter = '\t', index_col = False)
        #data = data[data['UE_type'] == 'uav']
        ue_x, ue_y, ue_z = data['ue_x'],data['ue_y'],data['ue_z']
        bs_x, bs_y, bs_z = data['bs_x'],data['bs_y'],data['bs_z']
        ue_tx_power = data['UE_TX_Power']
        dist = np.linalg.norm(np.column_stack((ue_x-bs_x, ue_y-bs_y, ue_z-bs_z)), axis =1)
        dist = dist[data['UE_type']=='uav']
        ue_tx_power = ue_tx_power[data['UE_type']=='uav']
        #SINR = data[data['UE_type']=='g_ue'][feature] #-noise
        SINR = data[feature] #-noise
        SINR_uav = SINR[data['UE_type']=='uav']
        SINR_gue = SINR[data['UE_type']=='g_ue']

        #SNR = data[feature2]
        #SNR_uav = SNR[data['UE_type']=='uav']
        #SNR_gue = SNR[data['UE_type'] == 'g_ue']

        #print (ue_tx_power[SINR>60], dist[SINR>60])
        plt.plot(np.sort(SINR), np.linspace(0, 1, len(SINR)), colors[j] + l_t[i],
                 label=' $\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))
        #plt.plot(np.sort(SINR), np.linspace(0,1,len(SINR)),colors[j]+l_t[i], label = 'SINR for all, $\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))
        #plt.plot(np.sort(SINR_uav), np.linspace(0, 1, len(SINR_uav)), colors[j] + '-.', label='SINR by UAV, $\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))
        #plt.plot(np.sort(SINR_gue), np.linspace(0, 1, len(SINR_gue)), colors[j] + ':', label='SINR by gUE, $\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))

        #plt.plot(np.sort(SNR), np.linspace(0, 1, len(SNR)), 'r' + l_t[i],label = 'SNR for all, $\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))
        #plt.plot(np.sort(SNR_uav), np.linspace(0, 1, len(SNR_uav)), 'r' + '-.', label='SNR by UAV, $\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))
        #plt.plot(np.sort(SNR_gue), np.linspace(0, 1, len(SNR_gue)), 'r' + ':', label='SNR by gUE, $\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))

        #label='SNR, $\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))
        #plt.title ( '$\\alpha$ = %s, $P_0$ = -%s' % (alpha, P0))

data = pd.read_csv('../data/ptr_full_power/uplink_interf_and_data_60_.txt',delimiter = '\t', index_col = False)
SINR = data[feature] #-noise
plt.plot(np.sort(SINR), np.linspace(0,1,len(SINR)), label = 'Full Power Control',color = 'k', lw = 3)
#ue_tx_power = ue_tx_power[data['UE_type']=='uav']
#SINR = data[data['UE_type']=='g_ue'][feature] #-noise
'''
SINR = data[feature]
SINR_uav = SINR[data['UE_type']=='uav']
SINR_gue = SINR[data['UE_type']=='g_ue']
SNR = data[feature2]
SNR_uav = SNR[data['UE_type']=='uav']
SNR_gue = SNR[data['UE_type'] == 'g_ue']

#print (ue_tx_power[SINR>60], dist[SINR>60])
color = 'tab:blue'
plt.plot(np.sort(SINR), np.linspace(0,1,len(SINR)),color,  label = 'SINR for all with full power')
plt.plot(np.sort(SINR_uav), np.linspace(0, 1, len(SINR_uav)), color, linestyle= '-.', label='SINR by UAV with full power')
plt.plot(np.sort(SINR_gue), np.linspace(0, 1, len(SINR_gue)), color , linestyle = ':', label='SINR by gUE with full power')
color = 'tab:red'
plt.plot(np.sort(SNR), np.linspace(0, 1, len(SNR)), color ,label = 'SNR for all with full power')
plt.plot(np.sort(SNR_uav), np.linspace(0, 1, len(SNR_uav)), color, linestyle = '-.', label='SNR by UAV with full power')
plt.plot(np.sort(SNR_gue), np.linspace(0, 1, len(SNR_gue)), color, linestyle = ':', label='SNR by gUE with full power')
'''
#plt.title('interference (dBm)')
plt.title('SINR (dB)')
#plt.title(' UE Tx Power (dBm)')
#plt.title ('INR (dB)')
plt.grid()
plt.legend()
plt.show()
