import numpy as np
import random
import pandas as pd
import sys
import pathlib
ab_path = pathlib.Path().absolute().parent.__str__()
print (ab_path)
sys.path.append(ab_path+'/uav_interference_analysis/')


def save_file(DATA, path):
    # function to save panda data
    random.seed(10)
    keys = DATA[0].keys()
    DATA_all = {key:[] for key in keys}
    for data in DATA:
        for key in keys:
            DATA_all[key].append(data[key])
    df = pd.DataFrame(DATA_all)
    #print(df['itf_no_ptx_list'])
    #power_optimizer = power_optimization(l_f = df['l_f'], ue_ids = df['ue_id'],
    #                                     itf_no_ptx_list= df['itf_no_ptx_list'],
    #                                     itf_id_list=df['itf_id_list'])
    #P_tx, SINR_list = power_optimizer.get_optimal_power_and_SINR(minstep=1e-7, debug=False)
    #df['tx_power_opt'] = 10*np.log10(P_tx)
    #df['SINR_opt'] = SINR_list

    df.to_csv(path, sep='\t', index=False)

def check_min_dist(loc):
    # check minimum distance between BSs
    I = np.argsort(loc[:,0])
    k =I[0]
    for i in I[1:]:
        dist = np.linalg.norm(loc[i] - loc[k])
        if dist <10:
            np.delete(loc, i, axis= 0)
        k = i
    return loc

def get_location(n=50, MAX = 1000, isBS = False, low =2, high = 5, h = 30 ):
    # set the locations of BSs
    loc_xy = np.random.uniform(low= 0, high =MAX, size =[n, 2])
    if isBS is True:
        height = np.random.uniform(low=low, high = high, size = (n,))
    else:
        height = np.repeat([h], n)
    loc = np.append(loc_xy, height[:,None], axis = 1)
    loc = check_min_dist(loc)

    return loc

def check_min_dist_UE_BS(bs_loc, g_UE_loc, n):
    n_gUE_dropped = n
    min_distance = 10
    # check minimum distance between ground UE and BS
    dist_vec = bs_loc - g_UE_loc[:,None]
    distance = np.linalg.norm(dist_vec, axis = -1)
    
    for j in range(n_gUE_dropped):
        min = np.min(distance[j])
        arg = np.argmin(distance[j])
        if min < min_distance:
            theta = np.random.uniform(low = - np.pi, high=np.pi, size = 1)
            d_h = bs_loc[arg,2]-g_UE_loc[j,2]
            r_2d = np.sqrt(min_distance**2 - d_h**2)
            # change the location of a UE so that the distance is more than minimum
            g_UE_loc[j, 0] = bs_loc[arg,0] + r_2d*np.cos(theta)
            g_UE_loc[j, 1] = bs_loc[arg,1] + r_2d*np.sin(theta)

    return g_UE_loc

