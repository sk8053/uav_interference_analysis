import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n',action='store',default=30,type= int,\
    help='number of iteration')
parser.add_argument('--f',action='store',default='SINR',type= str,\
    help='feature to plot')

parser.add_argument('--n_s',action='store',default=1,type= int,\
    help='number of streams')
parser.add_argument('--h',action='store',default=60,type= int,\
    help='UAV height')
#parser.add_argument('--f',action='store',default=28e9,type= float,\
#    help='frequency')

#plt.rcParams["font.family"] = "Times New Roman"
args = parser.parse_args()
n_iter = args.n
freq = 28e9
height = args.h
n_s= args.n_s
feature = args.f
plt.figure(figsize=(7,6))
KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6

KT_lin = 10**(0.1*KT)
NF_lin = 10**(0.1*NF)
power_control = False
colors = {200:'r',400:'b',800:'k', 0:'g'}
def get_df(dir_=None, isd_d = 200,  n_uav = 5, n_s = 1, n_path = 0, feature ='SINR', height =60, cb_bf = False
           ,cb_size = 64, uav=False, n_t =10):
    df = pd.DataFrame()
    t_n = np.array([])
   # print('../%s/uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_28G_%d.txt' % (dir_, n_uav, isd_d, n_s, height, 1))
    for t in range(n_iter):
        if cb_bf is False:

            data = pd.read_csv('../%s/uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_28G_%d.txt' % (
                dir_, n_uav, isd_d, n_s, height, t), delimiter='\t')
        else:
            dir_2 = '%d_path' % n_path
            #data = pd.read_csv('../%s/%s/cb_uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_28G_%d.txt' % (
            #    dir_,dir_2, n_uav, isd_d, n_s, height, t), delimiter='\t')
            data = pd.read_csv('../%s/cb_uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_28G_%d.txt' % (
                dir_,   n_uav, isd_d, n_s, height, t), delimiter='\t')
        df = pd.concat([df,data])


    #if uav is True:
    #    df0 = df[df['ue_type']=='uav']
    #else:
    #    df0 = df[df['ue_type'] == 'g_ue']
    df0 = df
    #df0 = df[df['bs_type']=='bs_d']
    print (len(df0), np.sum(df0['bs_type'] == 'bs_s'), isd_d)
    noise_power_dB = 10*np.log10(BW) + KT+NF
    noise_power_lin = KT_lin * NF_lin * BW


    tx_power = df0['tx_power']

    l_f = df0['l_f']
    intra_itf = df0['intra_itf']
    inter_itf = df0['inter_itf']

    if n_s ==1:
        intra_itf_lin = 0
    else:
        intra_itf_lin = 10 ** (0.1 * intra_itf)

    inter_itf_lin = 10**(0.1*inter_itf)
    total_itf_lin =   inter_itf_lin + intra_itf_lin
    noise_and_itf_dB = 10*np.log10(noise_power_lin + total_itf_lin)

    SINR = tx_power + l_f - noise_and_itf_dB

    SNR = tx_power + l_f - noise_power_dB
    INR = 10*np.log10(total_itf_lin) - noise_power_dB
    n_t = df0['n_t']
    capacity = (n_s / n_t) * BW * np.log2(1 + 10 ** (0.1 * SINR)) / 1e9
    bf_gain = df0['bf_gain']
    v_dict = {'SINR':SINR, 'SNR':SNR, 'INR':INR, 'Tx_power':df0['tx_power'], 'capacity':capacity, 'bf_gain':bf_gain}
    colors_= {64:'b', 128:'g', 256:'k', 512:'y', 1024:'m'}
    lt_ = {64:'--',128:':',256:'-.',512:'-'}
    colors_ = {1:'r', 2:'g', 10:'b'}
    #lt_ = {0:'-', 5:'-.'}
    if cb_bf is False:
        label = 'MMSE BF'
        color = 'k'
        line_style = '-'
    else:
        label = 'CB BF'# (n_path = %d)'%n_path
        #color = colors_[n_path]
        #line_style = lt_[cb_size]
        plt.title(  'beam forming gain' + r'(n$_s$ = %d)'%n_s + '\n codebook size = %d'%cb_size, fontsize = 18)
    plt.plot(np.sort(v_dict[feature]), np.linspace(0, 1, len(SINR)), lw=2.5, label= label)
                 #color=color, linestyle=line_style)



    return  np.array(SINR), SNR

#n_s4 = 4
n_s= 1
uav = False
closed = False

if closed is True:
    dir_1 = 'test_data_closed/%d_stream_ptrl'%n_s
    dir_2 = 'test_data_closed/%d_stream_ptrl' % (n_s+1)
    dir_3 = 'test_data_closed/%d_stream_ptrl' % (n_s+2)
    dir_4 = 'test_data_closed/%d_stream_ptrl' % (n_s+3)
else:
    dir_1 = 'test_data/%d_stream_ptrl' % n_s
    dir_2 = 'test_data/%d_stream_ptrl' % (n_s + 1)
    dir_3 = 'test_data/%d_stream_ptrl' % (n_s + 2)
    dir_4 = 'test_data/%d_stream_ptrl' % (n_s + 3)
#_, (ax1, ax2) = plt.subplots(1, 2)
#sinr_5, snr_400 = get_df(dir_ = dir_1, n_s = n_s, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120)

sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120)
#sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120, cb_bf = True, cb_size=64)
#sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120, cb_bf = True, cb_size=128)
#sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120, cb_bf = True, cb_size=512, n_path=1)
#sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120, cb_bf = True, cb_size=512, n_path = 2)
sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120, cb_bf = True, cb_size=512)
#sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120, cb_bf = True, cb_size=512)
#sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120, cb_bf = True, cb_size=1024)
#sinr_0, snr_400 = get_df(dir_ = dir_2, n_s = n_s+1, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120, cb_bf = True, cb_size=2048)

#sinr_400, snr_400 = get_df(dir_ = dir_3, n_s = n_s+2, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120)
#sinr_400, snr_400 = get_df(dir_ = dir_4, n_s = n_s+3, isd_d = 0, uav = uav, n_t = 10, feature= feature, height = 120)


if feature == 'SINR':
    plt.xlabel('SINR (dB)', fontsize = 16)
    plt.xlim(-10,42)
elif feature == 'SNR':
    plt.xlabel('SNR (dB)', fontsize=16)
    plt.xlim(-10, 52)

elif feature == 'INR':
    plt.xlabel('INR (dB)', fontsize=16)
    plt.xlim(-25, 22)
elif feature == 'Tx_power':
    plt.xlabel('Tx power (dBm)', fontsize = 16)
    plt.xlim([8,25])
elif feature == 'capacity':
    plt.xlabel ('Data Rate (Gbps)', fontsize = 16)
    plt.xlim([0, 2.5])
elif feature == 'bf_gain':
    plt.xlabel ('Beamforming gain (dB)', fontsize = 16)
    #plt.xlim(10,31)
#plt.xlabel ('Tx power', fontsize = 16)
#plt.xlim([-10,52])
#plt.xlim([7,24])

#plt.title ('UAV height = 60m', fontsize =16)
#sinr_400, snr_400 = get_df(dir_ = dir_2, n_s = 2, isd_d = 0, uav = uav, n_t = 5)
plt.legend(fontsize = 18)
#sinr_200, snr_200 = get_df(dir_ = dir_4, n_s = n_s, isd_d = 200, uav = True)
#sinr_400, snr_400 = get_df(dir_ = dir_2, n_s = n_s, isd_d = 400, uav = True)
#sinr_400, snr_400 = get_df(dir_ = dir_1, n_s = n_s, isd_d =800, uav  = True)
#sinr_400, snr_400 = get_df(dir_ = dir_1, n_s = n_s, isd_d = 0, uav = True)

#sinr_800, snr_800 = get_df(dir_ = dir_1, n_s = n_s, isd_d = 800)
#sinr_0, snr_0 = get_df(dir_ = dir_1, n_s = n_s, isd_d = 0)

plt.grid()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)


#plt.xlabel (r'SINR (dB), N$_s$ = %d, ISD$_s$ = %dm'% (n_s,200 ), fontsize = 16)
#plt.xlabel (r'Rate (bps), N$_s$ = %d, ISD$_s$ = %dm'% (n_s,200 ), fontsize = 16)
#plt.title ('Beamforming Gain Distribution, '+r'N$_s$ = %d'%n_s, fontsize = 16)
#plt.xlabel ('Beamforming Gain', fontsize = 16)
#plt.xlabel (r'Power (dBm), N$_s$ = %d, ISD$_s$ = %dm'% (n_s,200 ), fontsize = 16)
#plt.title ('UAV height = 60m', fontsize = 16)

#plt.ylabel('Probability of Coverage, P', fontsize = 16)
plt.ylabel('CDF ', fontsize = 16)
#plt.legend(fontsize = 15)
#plt.subplots_adjust(bottom=0.12,left = 0.1, top = 0.977, right = 0.97)

if closed is True:
    plt.savefig('uav_%s_120m.png'%feature, dpi = 500)
else:
    plt.savefig('uav_%s_120m.png'%feature, dpi = 500)

plt.show()



'''
lines =[]
if True:
    #lines += ax.plot(np.sort(df1_sinr), np.linspace(0,1,len(df1_sinr)),'b', label = 'UAV = 0 %', lw = 2)
    lines += ax.plot(np.sort(df2_sinr), np.linspace(0,1,len(df2_sinr)),'g', label = 'UAV = 12.5 %', lw = 2)
    #lines += ax.plot(np.sort(df3_sinr), np.linspace(0,1,len(df3_sinr)),'k', label = 'UAV = 25 %', lw =2)
    lines += ax.plot(np.sort(df4_sinr), np.linspace(0,1,len(df4_sinr)),'r', label = 'UAV = 37.5 %', lw=2)
    lines += ax.plot(np.sort(df5_sinr), np.linspace(0, 1, len(df5_sinr)), 'c', label='UAV = 50 %', lw=2)
if False:
    #lines+=  ax.plot(np.sort(df1_snr), np.linspace(0,1,len(df1_snr)),'b:', label = 'UAV = 0 %', lw = 2)
    lines += ax.plot(np.sort(df2_snr), np.linspace(0,1,len(df2_snr)),'g:', label = 'UAV = 12.5 %', lw = 2)
    #lines += ax.plot(np.sort(df3_snr), np.linspace(0,1,len(df3_snr)),'k:', label = 'UAV = 25 %', lw =2)
    lines += ax.plot(np.sort(df4_snr), np.linspace(0,1,len(df4_snr)),'r:', label = 'UAV = 37.5 %', lw=2)
    lines += ax.plot(np.sort(df5_snr), np.linspace(0, 1, len(df5_snr)), 'c:', label='UAV = 50 %', lw=2)



legend1 = ax.legend(fontsize =13)
legend2 = Legend(ax, [lines[2], lines[5]],['SINR', 'SNR'], fontsize = 13,
                 bbox_to_anchor=(-0.02,1.115), loc="upper left", ncol = 2, frameon= False)
ax.add_artist(legend2)

data = dict()
data['5'] = sinr_5
data['0'] = sinr_0
data = pd.DataFrame(data)
data.to_csv('sinr_5_0', sep='\t', index=False)
'''