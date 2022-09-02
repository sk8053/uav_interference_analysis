import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.legend import Legend
from matplotlib.patches import Ellipse
#import matplotlib
import argparse
#matplotlib.rc('xtick', labelsize=16)
#matplotlib.rc('ytick', labelsize=16)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n', action='store', default=30, type=int, \
                    help='number of iteration')
parser.add_argument('--f', action='store', default='SINR', type=str, \
                    help='feature to plot')

parser.add_argument('--n_s', action='store', default=1, type=int, \
                    help='number of streams')
parser.add_argument('--h', action='store', default=60, type=int, \
                    help='UAV height')
# parser.add_argument('--f',action='store',default=28e9,type= float,\
#    help='frequency')

plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"]

#from matplotlib import rcParams
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Tahoma']

args = parser.parse_args()
n_iter = args.n
freq = 28e9
height = args.h
n_s = args.n_s
feature = args.f

KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6

KT_lin = 10 ** (0.1 * KT)
NF_lin = 10 ** (0.1 * NF)
power_control = False
colors = {200: 'r', 400: 'b', 800: 'k', 0: 'g'}


def get_df(dir_=None, isd_d=200, n_uav=5, n_s=1, uav=False, n_t=10, feature='SINR', height=120, label_ = 'case 1',
           full_power = False, zero_tilted = False, ax = plt):
    df = pd.DataFrame()
    t_n = np.array([])

    for t in range(n_iter):
        if zero_tilted is True:
            data = pd.read_csv('/%s/full_power_zero_tilted_uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_28G_%d.txt' % (
                dir_, n_uav, isd_d, n_s, height, t), delimiter='\t')
        elif full_power is True:
            data = pd.read_csv('/%s/full_power_uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_28G_%d.txt' % (
                dir_, n_uav, isd_d, n_s, height, t), delimiter='\t')
        else:
            data = pd.read_csv('/%s/uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_28G_%d.txt' % (
            dir_, n_uav, isd_d, n_s, height, t), delimiter='\t')
        df = pd.concat([df, data])

    if uav is True:
        df0 = df[df['ue_type'] == 'uav']
    else:
        df0 = df[df['ue_type'] == 'g_ue']

    print(len(df), len(df0), np.sum(df0['link_state']==1)/len(df0))
    noise_power_dB = 10 * np.log10(BW) + KT + NF
    noise_power_lin = KT_lin * NF_lin * BW

    tx_power = df0['tx_power']

    l_f = df0['l_f']
    intra_itf = df0['intra_itf']
    inter_itf = df0['inter_itf']

    if n_s == 1:
        intra_itf_lin = 0
    else:
        intra_itf_lin = 10 ** (0.1 * intra_itf)

    inter_itf_lin = 10 ** (0.1 * inter_itf)
    total_itf_lin = inter_itf_lin + intra_itf_lin
    noise_and_itf_dB = 10 * np.log10(noise_power_lin + total_itf_lin)

    SINR = tx_power + l_f - noise_and_itf_dB

    SNR = tx_power + l_f - noise_power_dB
    INR = 10 * np.log10(total_itf_lin) - noise_power_dB
    capacity = (n_s / n_t) * BW * np.log2(1 + 10 ** (0.1 * SINR)) / 1e9
    bf_gain = df0['bf_gain']
    v_dict = {'SINR': SINR, 'SNR': SNR, 'INR': INR, 'Tx_power': df0['tx_power'], 'capacity': capacity,
              'bf_gain': bf_gain}

    colors_ = {'case 1': 'r', 'case 2': 'g', 'case 3':'b', 'case 4':'k'}
    lt_ = {'case 1': '-', 'case 2': '-.', 'case 3':':', 'case 4':'--'}
    #np.savetxt(label_+'_SINR.txt', SINR)
    #np.savetxt(label_+'_INR.txt', INR)
    #if uav is True:
    #    ax.plot(np.sort(v_dict[feature]), np.linspace(0, 1, len(SINR)), label=  label_,
    #             color=colors_[label_], linestyle=lt_[label_]
    #             , linewidth=2.5)
    #else:
    #    ax.plot(np.sort(v_dict[feature]), np.linspace(0, 1, len(SINR)), label = label_,
    #            color = colors_[label_],
    #              linestyle=lt_[label_]
    #             , linewidth=2.5)

    return v_dict[feature], SNR


n_s = 1
uav = False




#dir_1 = 'home/seongjoonkang/backup/test_data/%d_stream_ptrl' % n_s
dir_1 = 'home/seongjoonkang/uav_interference_analysis/test_data/%d_stream_ptrl'%n_s
#dir_2 = 'test_data/%d_stream_ptrl' % (n_s + 1)
#dir_3 = 'test_data/%d_stream_ptrl' % (n_s + 2)
#dir_4 = 'test_data/%d_stream_ptrl' % (n_s + 3)

fig = plt.figure()
ax = fig.add_subplot(111, label = '1')
#plt.setp(ax.get_xticklabels(), fontsize=16)
#plt.setp(ax.get_yticklabels(), fontsize=16)
#plt.setp(ax2.get_xticklabels(), fontsize=16)
#plt.setp(ax2.get_yticklabels(), fontsize=16)

feature = 'INR'
inr1, _ = get_df(dir_=dir_1, n_s=n_s, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 0, label_ = 'case 1')
inr2, _ = get_df(dir_=dir_1, n_s=n_s, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = 'case 2')
inr3, _ = get_df(dir_=dir_1, n_s=n_s, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = 'case 3', full_power= True)
#inr4, _ = get_df(dir_=dir_1, n_s=n_s, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = 'case 4' , zero_tilted=True)


feature = 'SINR'
sinr1, _ = get_df(dir_=dir_1, n_s=n_s, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 0, label_ = 'case 1')
sinr2, _ = get_df(dir_=dir_1, n_s=n_s, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = 'case 2')
sinr3, _ = get_df(dir_=dir_1, n_s=n_s, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = 'case 3', full_power= True)
#sinr4, _ = get_df(dir_=dir_1, n_s=n_s, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = 'case 4' , zero_tilted=True)

colors_ = {'config 1': 'r', 'config 2': 'g', 'config 3': 'b', 'case 4': 'k'}
lt_ = {'config 1': '-', 'config 2': '-.', 'config 3': ':', 'case 4': '--'}

for sinr, case in zip([sinr1, sinr2,  sinr3], ['config 1','config 2',  'config 3']):
    ax.plot(np.sort(sinr), np.linspace(0, 1, len(sinr)), label= case ,
                 color=colors_[case], linestyle= lt_[case], linewidth=2.5)
ax.set_xlabel ('SINR', fontsize = 18)
ax.grid()
ax.set_ylabel('CDF ', fontsize=18)

ax2 = fig.add_subplot(111, label = '2', frame_on = False)

ax2.set_xlabel ('INR', fontsize = 18)

for sinr, case in zip([inr1,inr2,  inr3], ['config 1','config 2',  'config 3']):
    ax2.plot(np.flip(np.sort(sinr)), np.linspace(0, 1, len(sinr)), label= case ,
                 color=colors_[case], linestyle= lt_[case], linewidth=2.5)

ax2.set_ylabel('CCDF', fontsize = 18)

ellipse1 = Ellipse(xy=(10.5, .65), width=3.3, height=0.23,edgecolor='darkmagenta', fc='None', lw=2, linestyle='--')
ax.arrow(10.5, 0.77,0,0.12, width = 0.05, head_width = 1.1, head_length = 0.045, color = 'darkmagenta')
ellipse2 = Ellipse(xy=(11.18, .25), width=2.5, height=0.19, edgecolor='firebrick', fc='None', lw=2, linestyle='--')
ax.arrow(11.18, 0.15,0.0,-0.09, width = 0.05, head_width = 1.1, head_length = 0.045, color = 'firebrick')
#ax2.plot([0,2],[0,23], linestyle ='-', lw = 1.5, color = 'darkmagenta')

ax.tick_params(axis = 'x', colors='firebrick', labelsize =18)
ax.tick_params(axis = 'y', colors='firebrick', labelsize =18)
ax.add_artist(ellipse1)
ax.add_artist(ellipse2)
ax.set_xlim(-5, 35)
ax.set_ylim(0,1)
ax.xaxis.label.set_color('firebrick')
ax.yaxis.label.set_color('firebrick')
ax.set_xticks(np.arange(-5,40, 5))

ax2.tick_params(axis = 'x', colors='darkmagenta')
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position('top')
ax2.yaxis.set_label_position('right')
ax2.tick_params(axis = 'x',colors = 'darkmagenta', labelsize = 18)
ax2.tick_params(axis = 'y',colors = 'darkmagenta', labelsize = 18)
ax2.set_ylim(0,1)
ax2.set_xlim(-30,15)
ax2.set_yticks(np.linspace(0,1,6))
#ax2.set_yticklabels(np.round(np.flip(np.linspace(0,1,6)), 1), size=16)
#ax2.set_xticks(np.arange(-30,20, 5))

ax2.xaxis.label.set_color('darkmagenta')
ax2.yaxis.label.set_color('darkmagenta')


plt.legend(loc = 'center right', fontsize=18)
plt.subplots_adjust(bottom=0.122,left = 0.113, top = 0.877, right = 0.896)
plt.savefig('sinr_inr_comp.png', dpi = 400)

plt.show()

