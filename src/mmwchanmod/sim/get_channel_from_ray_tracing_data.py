# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:46:23 2022

@author: seongjoon kang
"""

import numpy as np
from tqdm import tqdm
import sys
sys.path.append("C://Users/gangs/Downloads/Boston/Boston/uav_interference_analysis")

from src.mmwchanmod.sim.chanmod import MPChan
import pandas as pd

def get_channel_from_ray_tracing_data(dataframe:pd.DataFrame()):
    # build channel instance from the pre-defined csv file having certain format
    # one channel instance includes pathloss, delay, angle of arrival and departure
    # MPChan channel class is utilized to describe multi-path channel between a Tx and a Rx.
    
    # input: data frame corresponding to one link
    # output: MPChan class having all the multi-path compoenents
    
    n_path = int(dataframe['n_path'])
    dly_list = []
    pl_list =[]
    aoa_list = []
    aod_list = []
    zoa_list = []
    zod_list = []
    for i in range(n_path):
        dly_list.append(float(dataframe['delay_%d'%(i+1)]))
        pl_list.append(float(dataframe['path_loss_%d'%(i+1)]))
        
        aoa_list.append(float(dataframe['aoa_%d'%(i+1)]))
        aod_list.append(float(dataframe['aod_%d'%(i+1)]))
        
        zoa_list.append(float(dataframe['zoa_%d'%(i+1)]))
        zod_list.append(float(dataframe['zod_%d'%(i+1)]))
        
    chan = MPChan()
    chan.link_state = int(dataframe['link state'])
    chan.dly = dly_list
    chan.pl = pl_list
    chan.ang = np.zeros((n_path, MPChan.nangle), dtype=np.float32)
    
    chan.ang[:, MPChan.aod_phi_ind] = aod_list
    chan.ang[:, MPChan.aoa_phi_ind] = aoa_list
    
    chan.ang[:, MPChan.aod_theta_ind] = zod_list
    chan.ang[:, MPChan.aoa_theta_ind] = zoa_list
    
    return chan
        
