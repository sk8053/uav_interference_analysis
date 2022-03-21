import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--n',action='store',default=50,type= int,\
    help='number of iteration')
parser.add_argument('--f',action='store',default=28e9,type= float,\
    help='frequency')

args = parser.parse_args()
n_iter = args.n
freq = args.f

UAV_Height = 30
ISD = 200
#ratio = 50
n_UAV = 60

UE_power = 23
KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6

KT_lin = 10**(0.1*KT)
NF_lin = 10**(0.1*NF)
noise_power = KT_lin * NF_lin * BW

def get_df(dir_="data_with_3gpp_channel", freq = 28e9):
    df = pd.DataFrame()
    for t in range(n_iter):
        if freq == 28e9 or freq == 73e9 or freq ==140e9:
            data = pd.read_csv('../%s/data_%dm_height/uplink_interf_and_data_%dG_%d.txt' % (
                dir_, UAV_Height,int(freq/1e9), t), delimiter='\t')
        else:
            data = pd.read_csv('../data_with_3gpp_channel/data_%dm_height/uplink_interf_and_data_2G_%d.txt' % (
                UAV_Height, t), delimiter='\t')
        df = pd.concat([df,data])
        if dir_ == 'data_with_3gpp_channel_UAVonly':
            df = df[df['ue_type'] == 'uav']
        elif dir_ == 'data_with_3gpp_channel_gUEonly':
            df = df[df['ue_type']=='g_ue']
        else:
            df = df[df['ue_type'] == 'uav']

        intra_itf = df['intra_itf']
        inter_itf = df['inter_itf']
        total_itf = 10**(0.1*inter_itf) + 10**(0.1*intra_itf)
        total_itf = 10*np.log10(total_itf)
        SINR = df['SINR']

        SNR = df['SNR']
    return df, SINR, SNR, intra_itf, inter_itf, total_itf
#df = df[df['ue_type'] == 'uav']
#df = df[df['ue_z'] >29]
#ue_x, ue_y, ue_z = df['ue_x'], df['ue_y'], df['ue_z']
#bs_x, bs_y, bs_z = df['bs_x'], df['bs_y'], df['bs_z']
#dist_vect = np.column_stack((ue_x - bs_x, ue_y - bs_y, ue_z-bs_z))
#distance = np.linalg.norm(dist_vect, axis = 1)
df1, SINR1, SNR1, intra_itf1, inter_itf1, total_itf1 = get_df(dir_='data_with_3gpp_channel',freq = 28e9)
df2, SINR2, SNR2, intra_itf2, inter_itf2, total_itf2 = get_df(dir_='data_with_3gpp_channel_UAVonly',freq = 28e9)
df3, SINR3, SNR3, intra_itf3, inter_itf3, total_itf3 = get_df(dir_='data_with_3gpp_channel_gUEonly',freq = 28e9)

#df2, SINR2, SNR2, intra_itf2, inter_itf2 = get_df(freq = 73e9)

#df3, SINR3, SNR3, intra_itf3, inter_itf3 = get_df(freq = 140e9)
C_UAV =SINR2 #(BW/2)*np.log2(1+10**(0.1*SINR2))
C_gUE = SINR3 #(BW/2)*np.log2(1+10**(0.1*SINR3))
C =  SINR1 #BW*np.log2(1+10**(0.1*SINR1))

plt.figure()
plt.plot(np.sort(C), np.linspace(0,1,len(C)), label = 'Hybrid')
plt.plot(np.sort(C_UAV), np.linspace(0,1,len(C_UAV)), label = 'UAV only')
plt.plot(np.sort(C_gUE), np.linspace(0,1,len(C_gUE)), label = 'gUE only')
#plt.scatter (SINR3, intra_itf3)
#plt.plot(np.sort(SINR1), np.linspace(0,1,len(SINR1)),'r', label = 'SINR 28GHz')
#plt.plot(np.sort(SNR1), np.linspace(0,1,len(SNR1)),'k', label = 'SNR 28GHz')

#plt.plot(np.sort(SINR2), np.linspace(0,1,len(SINR2)),'r-.', label = 'SINR 73GHz')
#plt.plot(np.sort(SNR2), np.linspace(0,1,len(SNR2)),'k-.', label = 'SNR 73GHz')

#plt.plot(np.sort(SINR3), np.linspace(0,1,len(SINR3)),'r:', label = 'SINR 140GHz')
#plt.plot(np.sort(SNR3), np.linspace(0,1,len(SNR3)),'k:', label = 'SNR 140GHz')

plt.legend(fontsize =14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel('SNR and SINR (dB)', fontsize =14)
plt.ylabel('CDF', fontsize = 14)
plt.title ('SNR and SINR at %dm height for multi-UE case (%dGHz)'%(UAV_Height,int(freq/1e9)))
plt.grid()
#plt.savefig('snr_sinr_mmse_%dG_%dm_height_UAV=gUE=%d_ISD_%d.png'%(int(freq/1e9), UAV_Height,n_UAV,ISD ), dpi=500)


#bf1 = df1['tx_power']
#bf2 =  df2['tx_power']
#plt.figure()
#plt.plot(np.sort(bf1), np.linspace(0,1,len(bf1)), 'k-.', label = 'Tx Power without power ctrl')
#plt.plot(np.sort(bf2), np.linspace(0,1,len(bf2)), 'k', label = 'Tx Power with power ctrl')

#plt.scatter(distance, bf)
#plt.grid()
#plt.xticks(fontsize = 14)
#plt.yticks(fontsize = 14)
#plt.xlabel ("TX power distribution (dBm)")
#plt.ylabel('CDF', fontsize = 14)
#plt.title ('TX power distribution')
#plt.title ('Beamforming gain distribution by MMSE receiver')
#plt.savefig('bf_gain_mmse_%dGHz_%dm_height_UAV=gUE=%d_ISD_%d.png'%(int(freq/1e9),UAV_Height,n_UAV,ISD ), dpi = 1200)


#plt.figure()
#noise_power_db = 10*np.log10(noise_power)

#inr_intra1 = intra_itf1 - noise_power_db
#inr_inter1 = inter_itf1 - noise_power_db
#inr_tot = 10*np.log10(10**(0.1*intra_itf1) + 10**(0.1*inter_itf1)) - noise_power_db
#inr_intra2 = intra_itf2 - noise_power_db
#inr_inter2 = inter_itf2 - noise_power_db

#inr_intra3 = intra_itf3 - noise_power_db
#inr_inter3 = inter_itf3 - noise_power_db

#total_itf = intra_itf + inter_itf
#inr_total = 10*np.log10(total_itf) - noise_power_db



#plt.plot(np.sort(inr_tot), np.linspace(0,1,len(inr_tot)), 'r', label = 'total INR 28GHz')
#plt.plot(np.sort(inr_inter1), np.linspace(0,1,len(inr_inter1)), 'b', label = 'inter-INR 28GHz')

#plt.plot(np.sort(inr_intra2[I2]), np.linspace(0,1,len(inr_intra2[I2])), 'r-.', label = 'intra-INR 73GHz')
#plt.plot(np.sort(inr_inter2), np.linspace(0,1,len(inr_inter2)), 'b-.', label = 'inter-INR 73GHz')

#plt.plot(np.sort(inr_intra3[I3]), np.linspace(0,1,len(inr_intra3[I3])), 'r:', label = 'intra-INR 140GHz')
#plt.plot(np.sort(inr_inter3), np.linspace(0,1,len(inr_inter3)), 'b:', label = 'inter-INR 140GHz')
#plt.plot(np.sort(inr_total), np.linspace(0,1,len(inr_total)), 'k', label = 'total interference/noise')
#plt.xticks(fontsize = 14)
#plt.yticks(fontsize = 14)
#plt.xlabel('INR (dB)', fontsize =14)
#plt.ylabel('CDF', fontsize = 14)
#plt.title ('INR comparison (%dGHz)'%(int(freq/1e9)))
#plt.legend()
#plt.grid()
#plt.savefig('INR_mmse_%dGHz_%dm_height_UAV=gUE=%d_ISD_%d.png'%(int(freq/1e9), UAV_Height,n_UAV,ISD ), dpi = 500)
plt.show()

