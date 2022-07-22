import numpy as np
import pandas as pd
from tqdm import tqdm
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from mmwchanmod.sim.chanmod_old import MPChan, dir_path_loss_multi_sect
from mmwchanmod.sim.chanmod_old import MPChan

thetabw_, phibw_ = 65, 65 # half power of beamwidth
frequency = 28e9
n_sect = 3 # number of sector of BS
UE_TX_power = 23

KT = -174
NF = 6
BW = 400e6

# ue antenna configuration
element_ue = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
arr_ue = URA(elem=element_ue, nant=np.array([4, 4]), fc=frequency)
arr_ue = RotatedArray(arr_ue, theta0=-90)
arr_ue_list = [arr_ue]

# gnb antenna configuration
elem_gnb_t = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8, 8]), fc=frequency)
arr_gnb_list = multi_sect_array(arr_gnb_t, sect_type='azimuth', nsect=n_sect, theta0=-12)

data = pd.read_csv('paths.csv')
SNR_list = []
no_MIMO = True
for j in tqdm(range (len(data))):
    sample_data = data.iloc[j,:]
    chan = MPChan()
    pl= []
    aoa, aod = [],[]
    zoa, zod = [], []
    n_path = int(sample_data[3])
    # 1) pathloss 2) zenith angle of arrival 3) azimuth angle of arrival
    # 4) zenith angle of departure 5) azimuth angle of departure
    for i in range (n_path):
        pl.append(-sample_data[5*i + 4])
        # we don't need delay-component to consider channel
        #dly.append(sample_data[6*i +2])
        zoa.append(sample_data[5*i +5])
        aoa.append(sample_data[5*i +6])
        zod.append(sample_data[5*i +7])
        aod.append(sample_data[5*i +8])
    chan.pl = pl
    #chan.dly = dly # we don't need delay-component to consider channel
    chan.ang = np.zeros((n_path, MPChan.nangle), dtype=np.float32)
    chan.ang[:, MPChan.aoa_phi_ind] = aoa
    chan.ang[:,MPChan.aod_phi_ind] = aod
    chan.ang[:, MPChan.aoa_theta_ind] = zoa
    chan.ang[:, MPChan.aod_theta_ind] = zod

    # let's assume that every link is NLOS
    chan.link_state =2
    if no_MIMO is True:
        pl_gain = 10*np.log10(np.sum(10**(-0.1*np.array(pl))));
    else:
        data_i = dir_path_loss_multi_sect(arr_gnb_list, arr_ue_list, chan,long_term_bf=True)
        # get path-gains including beamforming and element gains
        pl_gain = data_i['pl_eff']
    SNR =UE_TX_power - np.array(pl_gain) - KT -NF - 10 * np.log10(BW)
    SNR_list.append(SNR)

new_data = dict()
new_data['SNR'] = SNR_list
new_data = pd.DataFrame(new_data)
new_data.to_csv('snr_data_no_MIMO.csv')

