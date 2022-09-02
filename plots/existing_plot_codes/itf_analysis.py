import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--n',action='store',default=30,type= int,\
    help='number of iteration')
parser.add_argument('--f',action='store',default=28e9,type= float,\
    help='frequency')

args = parser.parse_args()
n_iter = args.n
freq = args.f


ISD = 200
#ratio = 50
#n_UAV = 60

UE_power = 23
KT = -174
NF = 6
if freq == 28e9:
    BW = 400e6
else:
    BW = 80e6
dir_ = "data_with_nn_1_marco"
UAV_Height = 120

KT_lin = 10**(0.1*KT)
NF_lin = 10**(0.1*NF)
#def get_df(UAV_Height, dir_=None):
df = pd.DataFrame()

itf_UAV_map = dict()
for t in range(n_iter):
    if freq == 28e9:
        data = pd.read_csv('../%s/data_%dm_height/uplink_interf_and_data_28G_%d.txt' % (
            dir_, UAV_Height,  t), delimiter='\t')
    else:
        data = pd.read_csv('../data_with_3gpp_channel/data_%dm_height/uplink_interf_and_data_2G_%d.txt' % (
            UAV_Height, t), delimiter='\t')
    #df = pd.concat([df,data])
    itf_UAV = np.array([])
    for i in np.arange(len(data)):
        data['itf_UAV'][i] = data['itf_UAV'][i][1:-1].split(',')

        data['itf_UAV'][i] = np.array(data['itf_UAV'][i], dtype = float)

        itf_UAV = np.append(itf_UAV,data['itf_UAV'][i])
    itf_UAV_map[str(t)] = itf_UAV

print (itf_UAV_map)


