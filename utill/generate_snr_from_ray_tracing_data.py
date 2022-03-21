import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
sys.path.append('src/')
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss_multi_sect
from mmwchanmod.sim.chanmod import MPChan

thetabw_, phibw_ = 65, 65 # half power of beamwidth
frequency = 28e9
n_sect = 3
UE_TX_power = 23

KT = -174
NF = 6
BW = 400e6

element_ue = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
arr_ue = URA(elem=element_ue, nant=np.array([4, 4]), fc=frequency)

arr_ue = RotatedArray(arr_ue, theta0=-90)
arr_ue_list = [arr_ue]

elem_gnb_t = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8, 8]), fc=frequency)
arr_gnb_list = multi_sect_array(arr_gnb_t, sect_type='azimuth', nsect=n_sect, theta0=0)
data3 = pd.read_csv('path_grid_3.csv')
data4 = pd.read_csv('path_grid_4.csv')
#print (len(data2))
SNR_list = []
for j in tqdm(range (len(data4))):
    sample_data = data4.iloc[j,:]
    chan = MPChan()
    pl= []
    dly = []
    aoa, aod = [],[]
    zoa, zod = [], []
    n_path = sample_data['n_path']
    for i in range (n_path):
        pl.append(sample_data['path_loss_%d'%(i+1)])
        dly.append(sample_data['delay_%d'%(i+1)])
        aoa.append(sample_data['aoa_%d'%(i+1)])
        aod.append(sample_data['aod_%d'%(i+1)])
        zoa.append(sample_data['zoa_%d' % (i + 1)])
        zod.append(sample_data['zod_%d' % (i + 1)])
    chan.pl = pl
    chan.dly = dly
    chan.ang = np.zeros((n_path, MPChan.nangle), dtype=np.float32)
    chan.ang[:, MPChan.aoa_phi_ind] = aoa
    chan.ang[:,MPChan.aod_phi_ind] = aod
    chan.ang[:, MPChan.aoa_theta_ind] = zoa
    chan.ang[:, MPChan.aod_theta_ind] = zod
    chan.link_state = sample_data['link state']
    data = dir_path_loss_multi_sect(arr_gnb_list, arr_ue_list, chan,
                                    long_term_bf=True, instantaneous_bf=False, isdrone= True)
    pl_gain = data['pl_eff']
    SNR =UE_TX_power - np.array(pl_gain) - KT -NF - 10 * np.log10(BW)
    SNR_list.append(SNR)

data4['SNR'] = SNR_list
data4.to_csv('data_4_copy.csv')

