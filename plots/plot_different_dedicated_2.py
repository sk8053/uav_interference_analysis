import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.patches import Ellipse
import argparse

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
    n_t = df0['n_t']
    capacity = (n_s / n_t) * BW * np.log2(1 + 10 ** (0.1 * SINR)) / 1e9
    bf_gain = df0['bf_gain']
    v_dict = {'SINR': SINR, 'SNR': SNR, 'INR': INR, 'Tx_power': df0['tx_power'], 'capacity': capacity,
              'bf_gain': bf_gain}

    colors_ = {r'n$_s$ = 1': 'r', r'n$_s$ = 2': 'g', 'case 3':'b', 'case 4':'k'}
    lt_ = {'case 1': '-', 'case 2': '-.', 'case 3':':', 'case 4':'--'}



    return v_dict[feature], SNR


n_s = 1
uav = True
closed = False
abs_dir = 'home/seongjoonkang/backup/'
if closed is True:
    dir_1 = abs_dir + 'test_data_closed/%d_stream_ptrl' % n_s
    dir_2 = abs_dir + 'test_data_closed/%d_stream_ptrl' % (n_s + 1)
    dir_3 = abs_dir + 'test_data_closed/%d_stream_ptrl' % (n_s + 2)
    dir_4 = abs_dir + 'test_data_closed/%d_stream_ptrl' % (n_s + 3)
else:
    dir_1 = abs_dir + 'test_data/%d_stream_ptrl' % n_s
    dir_2 = abs_dir + 'test_data/%d_stream_ptrl' % (n_s + 1)
    dir_3 = abs_dir + 'test_data/%d_stream_ptrl' % (n_s + 2)
    dir_4 = abs_dir + 'test_data/%d_stream_ptrl' % (n_s + 3)

fig = plt.figure()
ax = fig.add_subplot(111, label = '1')

plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

feature = 'SINR'
sinr1, _ = get_df(dir_=dir_2, n_s=2, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = r'n$_s$ = 1')
sinr2, _ = get_df(dir_=dir_2, n_s=2, isd_d=800, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = r'n$_s$ = 2')
sinr3, _ = get_df(dir_=dir_2, n_s=2, isd_d=400, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = r'n$_s$ = 3')
sinr4, _ = get_df(dir_=dir_2, n_s=2, isd_d=200, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = r'n$_s$ = 4')


feature = 'capacity'
rate1, _ = get_df(dir_=dir_2, n_s=2, isd_d=0, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = r'n$_s$ = 1')
rate2, _ = get_df(dir_=dir_2, n_s=2, isd_d=800, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = r'n$_s$ = 2')
rate3, _ = get_df(dir_=dir_2, n_s=2, isd_d=400, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = r'n$_s$ = 3')
rate4, _ = get_df(dir_=dir_2, n_s=2, isd_d=200, uav=uav, n_t=10, feature=feature, height=120, n_uav = 5, label_ = r'n$_s$ = 4')

colors_ = ['r','g','b','k']
lt_ = ['--',':',  '-.','-']
h_ = [0, 800, 400, 200]

for i, (rate, case) in enumerate(zip([rate1, rate2, rate3, rate4], [r'ISD$_d$ = %s'%('\u221e'),r'ISD$_d$ = 800m', r'ISD$_d$ = 400m', r'ISD$_d$ = 200m'])):
    if closed is True and i ==0:
        continue
    else:
        ax.plot(np.sort(rate), np.linspace(0, 1, len(rate)), label= case ,
                 color=colors_[i], linestyle= lt_[i], linewidth=2.5)
        np.savetxt('UAV_rate_open_access_ISD_d='+str(h_[i]), rate)


ax.set_xlabel ('Data Rate (Gbps)', fontsize = 18)
ax.grid()
ax.set_ylabel('CDF ', fontsize=18)

ax2 = fig.add_subplot(111, label = '2', frame_on =False)

ax2.set_xlabel ('SINR (dB)', fontsize = 18)
ax2.set_ylabel('CCDF ', fontsize=18)

for i, (sinr, case) in enumerate(zip([sinr1, sinr2, sinr3, sinr4], [r'ISD$_{\rm d}$ = %s'%('\u221e'),r'ISD$_{\rm d}$ = 800m', r'ISD$_{\rm d}$ = 400m', r'ISD$_{\rm d}$ = 200m'])):
    if closed is True and i ==0:
        continue
    else:
        ax2.plot(np.flip(np.sort(sinr)), np.linspace(0, 1, len(sinr)), label= case ,
                 color=colors_[i], linestyle= lt_[i], linewidth=2.5)
        np.savetxt('UAV_SINR_open_access_ISD_d='+str(h_[i]), sinr)
if closed is False:
    if uav is True:
        ellipse1 = Ellipse(xy=(12.5, .76), width=3.3, height=0.25,
                           edgecolor='darkmagenta', fc='None', lw=2, linestyle='--')
        ax2.arrow(12.5, 0.88, 0, 0.07, width=0.05, head_width=1.1, head_length=0.045, color='darkmagenta')

        ellipse2 = Ellipse(xy=(-0.05, .25), width=8.0, height=0.19,
                           edgecolor='firebrick', fc='None', lw=2, linestyle='--')
        ax2.arrow(-0.05, 0.155, 0.0, -0.09, width=0.05, head_width=1.1, head_length=0.045, color='firebrick')

    else:
        ellipse1 = Ellipse(xy=(12.5, .59), width=3.3, height=0.33,
                                edgecolor='darkmagenta', fc='None', lw=2, linestyle='--')
        ax2.arrow(12.5, 0.75,0,0.18, width = 0.05, head_width = 1.1, head_length = 0.045, color = 'darkmagenta')

        ellipse2 = Ellipse(xy=(-2.2, .25), width=2.5, height=0.19,
                                edgecolor='firebrick', fc='None', lw=2, linestyle='--')
        ax2.arrow(-2.2, 0.15,0.0,-0.09, width = 0.05, head_width = 1.1, head_length = 0.045, color = 'firebrick')
else:
    if uav is True:

        ellipse1 = Ellipse(xy=(11.5, .5), width=3.3, height=0.65,
                           edgecolor='darkmagenta', fc='None', lw=2, linestyle='--')
        ax2.arrow(11.5, 0.83, 0, 0.1, width=0.05, head_width=1.1, head_length=0.045, color='darkmagenta')

        ellipse2 = Ellipse(xy=(-20.05, .25), width=10.0, height=0.19,
                           edgecolor='firebrick', fc='None', lw=2, linestyle='--')
        ax2.arrow(-20.05, 0.155, 0.0, -0.09, width=0.05, head_width=1.1, head_length=0.045, color='firebrick')

    else:
        ellipse1 = Ellipse(xy=(7.5, .75), width=3.3, height=0.25,
                           edgecolor='darkmagenta', fc='None', lw=2, linestyle='--')
        ax2.arrow(7.5, 0.87, 0, 0.07, width=0.05, head_width=1.1, head_length=0.045, color='darkmagenta')

        ellipse2 = Ellipse(xy=(-18.05, .25), width=5.0, height=0.19,
                           edgecolor='firebrick', fc='None', lw=2, linestyle='--')
        ax2.arrow(-18.05, 0.155, 0.0, -0.09, width=0.05, head_width=1.1, head_length=0.045, color='firebrick')

ax2.tick_params(axis = 'x', colors='darkmagenta')
ax2.xaxis.label.set_color('darkmagenta')

ax.tick_params(axis = 'x', colors='firebrick')
ax.xaxis.label.set_color('firebrick')
ax2.add_artist(ellipse1)
ax2.add_artist(ellipse2)

ax.tick_params(axis = 'x', colors='firebrick', labelsize =18)
ax.tick_params(axis = 'y', colors='firebrick', labelsize =18)
ax.set_ylim(0,1)
ax.xaxis.label.set_color('firebrick')
ax.yaxis.label.set_color('firebrick')


ax2.tick_params(axis = 'x', colors='darkmagenta')
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position('top')
ax2.yaxis.set_label_position('right')
ax2.tick_params(axis = 'x',colors = 'darkmagenta', labelsize = 18)
ax2.tick_params(axis = 'y',colors = 'darkmagenta', labelsize = 18)
ax2.set_ylim(0,1)
#ax2.set_yticks(np.linspace(0,1,6))
#ax2.set_yticklabels(np.round(np.flip(np.linspace(0,1,6)), 1), size=16)

ax2.xaxis.label.set_color('darkmagenta')
ax2.yaxis.label.set_color('darkmagenta')

if closed is True:
    ax2.set_xlim(-25,35)
    ax.set_xlim(0,2.0)
    #ax.set_xticks(np.arange(0, 2.5, 0.5))
    #ax2.set_xticks(np.arange(-25, 40, 5))


else:
    ax2.set_xlim(-5, 35)
    ax.set_xlim(0, 4.0)
    #ax.set_xticks(np.arange(0, 4.5, 0.5))
    #ax2.set_xticks(np.arange(-5, 40, 5))

plt.legend(loc = 'center right', fontsize=18)
plt.subplots_adjust(bottom=0.127,left = 0.115, top = 0.868, right = 0.897)


if uav is True:
    if closed is True:
        plt.savefig('closed_uav_%s_120m_with_dedicated_v2.png'%feature, dpi = 400)
    else:
        plt.savefig('open_uav_%s_120m_with_dedicated_v2.png'%feature, dpi = 400)
else:
    if closed is True:
        plt.savefig('closed_gue_%s_120m_with_dedicated_v2.png'%feature, dpi = 400)
    else:
        plt.savefig('open_gue_%s_120m_with_dedicated_v2.png'%feature, dpi = 400)
plt.show()

