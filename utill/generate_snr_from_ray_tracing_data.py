import numpy as np
import pandas as pd
from tqdm import tqdm



UE_TX_power = 23
KT = -174
NF = 6
BW = 400e6

data = pd.read_csv('paths.csv')
SNR_list = []

for j in tqdm(range (len(data))):
    sample_data = data.iloc[j,:]
    pl= []
    n_path = int(sample_data[4])
    for i in range (n_path):
        pl.append(-sample_data[5*i + 5])

    pl_gain = 10*np.log10(np.sum(10**(-0.1*np.array(pl))));

    SNR =UE_TX_power + np.array(pl_gain) - KT -NF - 10 * np.log10(BW)
    SNR_list.append(SNR)

new_data = dict()
new_data['x'] = data.iloc[:,1]
new_data['y'] = data.iloc[:,2]
new_data['SNR'] = SNR_list
new_data = pd.DataFrame(new_data)
new_data.to_csv('snr_data_no_MIMO.csv')

