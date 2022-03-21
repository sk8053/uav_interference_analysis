import random

import sys
import pathlib
ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path+'/uav_interference_analysis/src/')

import numpy as np
from tqdm.auto import tqdm
from ue import  UE
from bs import BS
from network_channel import Channel_Info, Network
#import pandas as pd
from ground_channel_generation import GroundChannel
from mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
import argparse
from wrap_around import wrap_around
#from power_optimizer import power_optimization
from utills import check_min_dist_UE_BS, get_location, save_file

parser = argparse.ArgumentParser(description='')
parser.add_argument('--height',action='store',default=30,type= int,\
    help='uav_height')
parser.add_argument('--t_gpp', dest = 't_gpp', action = 'store_true')

parser.add_argument('--n_drop',action='store',default=100,type= int,\
    help='number of dropped UAV and ground UEs')
parser.add_argument('--ptrl', dest = 'ptrl', action = 'store_true')

parser.add_argument('--run_time',action='store',default=50,type= int,\
    help='simulation running time')
parser.add_argument('--frequency',action='store',default=28e9,type= float,\
    help='carrier frequency')
parser.add_argument('--alpha',action='store',default=0.5,type= float,\
    help='alpha for power control')
parser.add_argument('--ratio',action='store',default=0.5,type= float,\
    help='ratio of number')
parser.add_argument('--P0',action='store',default=-80,type= float,\
    help='reference power for power control')
parser.add_argument('--n', action ='store', default = 2, type = float,\
                    help = 'number of connections to each BS')

args = parser.parse_args()
UAV_height = args.height
# running time
run_time = args.run_time
alpha = args.alpha
P0 = args.P0
ratio = args.ratio
frequency = args.frequency
ptrl = args.ptrl
enable_3gpp_ch = args.t_gpp
max_n = args.n

if max_n>=1.0:
    max_n = int(max_n)

print ('number of connections to BS is ', max_n)
print ('frequency is ', frequency)
#print ('alpha is %s and P0 is %s'% (alpha, P0))

n_drop = args.n_drop

g_UE_Height = 1.6
print ('UAV height is ', UAV_height)
# set some parameters for network area
random.seed(100)
ISD = 200

print ('ISD is ,', ISD)
lam = 4/(np.pi*ISD**2)

XMAX, YMAX = 1000, 1000
area = XMAX * YMAX
print ('Average number of BS is ', int(lam*area))
min_distance = 10
# set network we observe
network = Network(X_MAX = XMAX, X_MIN = 0, Y_MAX = YMAX, Y_MIN = 0, Z_MAX = UAV_height)
# channel information

print ('Aerial channel for 3GPP is, ', enable_3gpp_ch)
channel_info = Channel_Info(three_gpp = enable_3gpp_ch)
channel_info.alpha = alpha
channel_info.P0 = P0
channel_info.ptrl = ptrl

channel_info.frequency = frequency
channel_info.long_term_bf = True
drone_antenna_gain = drone_antenna_gain()
print ('long-term beamforming is ', channel_info.long_term_bf)
if frequency == 2e9:
    channel_info.BW = 80e6

n_UAV_dropped, n_gUE_dropped =n_drop, n_drop # int(n_drop* n_UAV/(n_UAV+n_gUE)), int(n_drop* n_gUE/(n_UAV+n_gUE))
print('number of UAV and gUE are %d and %d'% (n_UAV_dropped, n_gUE_dropped))
print ('numbers of connected UAVs and UEs, %d and %d' %(max_n, max_n))

# every drop we set different number of BSs and their locations
# number of ground UE = number of BS, and their locations are closer
BS_DATA = []
random.seed(10)
for t in tqdm(range (run_time), desc= 'number of drops'):
    n_BS = np.random.poisson(lam*area)
    #N_g_UE = np.random.poisson(lam_ue_uav*area)

    channel_info.N_UAV = n_drop
    bs_loc_t = get_location(n = n_BS, MAX = 1000, isBS = True, low = 2, high = 5)
    heights_bs_a = np.random.uniform(low=10, high = 30 , size = (n_BS,))
    bs_loc_a = np.column_stack((bs_loc_t[:,:2], heights_bs_a))
    bs_loc = np.append(bs_loc_t, bs_loc_a, axis = 0)

    # generate all UAV locations we will utilize to drop over simulation time
    UAV_loc = get_location(n = n_UAV_dropped, MAX = 1000, isBS = False, h = UAV_height)
    # apply wrap_around between BS and g_UE
    #UAV_loc = bs_loc + wrap_around(UAV_loc - bs_loc, network)

    # set the locations of ground UEs
    # we have to randomize the xy location of UEs
    g_UE_loc = get_location(n = n_UAV_dropped, MAX = 1000, isBS = False, h = g_UE_Height)
    g_UE_loc = check_min_dist_UE_BS(bs_loc, g_UE_loc, n_gUE_dropped)

    # generate ground channels for all UEs
    # repeat g_UE locations by number of BSs
    g_UE_loc2 = np.repeat(g_UE_loc[None,:],len(bs_loc), axis = 0).reshape(-1,3)
    # repeat BS locations for all g_UEs
    bs_loc2 = np.repeat(bs_loc, n_gUE_dropped, axis=0)

    # apply wrap_around between BS and g_UE
    g_UE_loc2 = bs_loc2 + wrap_around(g_UE_loc2 - bs_loc2, network)

    g_channels = GroundChannel(tx=bs_loc2, rx = g_UE_loc2, aerial_fading = False, frequency = frequency)
    data = open(g_channels.path + 'ray_tracing.txt', 'r')

    g_channel_list =g_channels.getList(data)
    g_channel_list = np.array(g_channel_list).reshape(n_gUE_dropped,-1)

    BS_set = []
    # firstly drop BSs
    bs_id = 0
    for a in range(n_BS):
        # deploy terrestrial and aerial BSs
        bs_t = BS(network = network, bs_type=1, bs_id=bs_id, channel_info = channel_info, loc =bs_loc_t[a][:,None]) # 1 is terrestrial, 0 is aerial
        bs_a = BS(network=network, bs_type=0, bs_id=bs_id+1, channel_info=channel_info, loc=bs_loc_a[a][:, None])  # 1 is terrestrial, 0 is aerial
        BS_set.append(bs_t)
        BS_set.append(bs_a)
        bs_id += 2
    # secondly drop UAVs
    uav_list = []
    g_ue_list = []
    for k in np.arange(n_UAV_dropped):
        uav = UE(channel_info=channel_info, network=network, UE_id=k, ground_UE=False, loc=UAV_loc[k], bs_info=BS_set,
                 drone_antenna_gain = drone_antenna_gain, enable_3gpp_ch = enable_3gpp_ch)
        uav.association()
        uav_list.append(uav)
    # drop g UEs
    for k in np.arange(n_gUE_dropped):
        g_UE = UE(channel_info=channel_info, network=network, bs_info=BS_set,  UE_id=k + n_UAV_dropped,
                   ground_UE=True, loc=g_UE_loc[k])
        g_UE.set_ground_user_channel(chan_list=g_channel_list[k])
        g_UE.association()  # drop ground UE
        g_ue_list.append(g_UE)

    total_ue_list = np.append(uav_list, g_ue_list)
    # decide the association and allocate resources
    total_connected_ue_id_list = []
    for bs in BS_set:
        if bs.connected is True:
            connection_complete, connected_ue_id = bs.decide_connection_bt_uav_ue(max_n=max_n)
            if connection_complete is True:
                total_connected_ue_id_list = np.append(total_connected_ue_id_list, connected_ue_id)

    UL_DATA = np.array([])
    connected_BS_list = np.array([])
    for bs in BS_set:
        if bs.connected is True:
            data = bs.get_interference(total_UE_list= total_ue_list, total_connected_ue_id_list = total_connected_ue_id_list)
            UL_DATA = np.append(UL_DATA, data)
            connected_BS_list = np.append(connected_BS_list, bs)

    DL_DATA = np.array([])
    connected_uav_ids = total_connected_ue_id_list[total_connected_ue_id_list<n_UAV_dropped]
    connected_uavs= total_ue_list[np.array(connected_uav_ids, dtype = int)]
    #connected_ue_ids = total_connected_ue_id_list[total_connected_ue_id_list >= n_UAV_dropped]
    #connected_ues = total_ue_list[connected_ue_ids]
    for ue_id in total_connected_ue_id_list:
        ue = total_ue_list[int(ue_id)]
        if ue.get_ue_type() == 'g_ue':
            itf_ue_uav = ue.get_interference_from_uav(connected_uavs = connected_uavs)
        else:
            itf_ue_uav = -200

        data = ue.get_interference(connected_BS = connected_BS_list)
        data['itf_ue_uav'] = itf_ue_uav
        DL_DATA = np.append(DL_DATA, data)



    # collect all data and save from only UAVs
    path_ul = ab_path + '/uav_interference_analysis/test_data/dedicated_%d_%d_ptrl/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                  (max_n, max_n, max_n, UAV_height,  int(frequency / 1e9), t)
    path_dl = ab_path + '/uav_interference_analysis/test_data/dedicated_%d_%d_ptrl/downlink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                  (max_n, max_n, max_n, UAV_height,  int(frequency / 1e9), t)
    save_file(UL_DATA, path_ul)
    save_file(DL_DATA, path_dl)

