# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:43:56 2021

@author: sdran
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pwc_opt import UtilMax

idrop = 0
ii=50
data_dir = 'data_60/data_%d_2'%ii

# Parameters
tx_pow = 23
nf = 6
bw = 400e6
EkT = -174
ntrials = 30
ndrops = 30

rate_list = [np.zeros(0), np.zeros(0)]
g0_list =[]
sinr_list_opt, sinr_list_pmax = [],[]
for it in range(ntrials):

    # Select a random drop
    #idrop = np.random.randint(0, ndrops, 1)
    idrop = it
    # Print status
    #print('trial = %d, drop=%d' % (it, idrop))

    # Read data
    pl_path = os.path.join(data_dir, 'pl_eff_mat/pl_eff_matrix_%d.csv' % idrop)
    df_pl = pd.read_csv(pl_path, sep=' ', header = None)
    nue, nbs = df_pl.shape
    pl = np.array(df_pl)

    # Find the index of the  serving BS for each UE from the file, ue_n.csv
    ue_ids = os.path.join(data_dir, 'ue_loc/ue_%d.csv' % idrop)
    df_ue = pd.read_csv(ue_ids, delimiter = '\t')

    # read UE id and BS id from csv file after association
    ue_ind = np.array(df_ue['ue_id'])
    bs_ind = np.array(df_ue['bs_id'])

    # Create a matrix G and vector g0 such that the vector of SNRs is
    #   snr = g0*p / (G*p + 1)
    nlinks = len(ue_ind)
    g0 = np.zeros(nlinks)
    G = np.zeros((nlinks, nlinks))

    GdB = tx_pow - nf - 10 * np.log10(bw) - EkT - pl
    Glin = 10 ** (0.1 * GdB)

    for i in range(nlinks):
        iue = ue_ind[i]
        ibs = bs_ind[i]

        g0[i] = Glin[iue, ibs]

        G[i, :] = Glin[iue, bs_ind]
        G[i, i] = 0
    g0_list = np.append(g0_list, 10*np.log10(g0))
    # Run the optimization
    util_max = UtilMax(g0, G)
    popt = util_max.grad_des_opt()

    # Add the rates to the list
    pmax = np.ones(nlinks)
    for i, p in enumerate([pmax, popt]):
        sinr, rate, util, ugrad = util_max.util_eval(p, \
                                  return_snr_rate=True)
        if i ==0:
            sinr_list_pmax = np.append(sinr_list_pmax, 10*np.log10(sinr))
        else:
            sinr_list_opt = np.append(sinr_list_opt, 10*np.log10(sinr))
        rate_list[i] = np.hstack((rate_list[i], rate))
np.savetxt('rate_%d.txt'%ii, rate_list)
# Plot the CDF of the rates over all the links
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

colors = ['r', 'k']
for i,rate in enumerate(rate_list):
    rate = np.sort(rate)*bw/1e6
    n = len(rate)
    ax2.plot(rate, np.arange(n) / n, colors[i])
    #plt.semilogx(rate, np.arange(n) / n)


ax2.set_xlabel('Capacity (Mbps)')
ax2.set_ylabel('CDF')
ax2.legend(['Capacity by Max Power', 'Capacity by Opt Power'], loc = 'lower right')

ax1.plot(np.sort(sinr_list_pmax), np.linspace(0,1,len(sinr_list_pmax)),'r-.')
ax1.plot(np.sort(sinr_list_opt), np.linspace(0,1,len(sinr_list_opt)),'k-.')
ax1.plot (np.linspace(-10, 68, 60), np.repeat(0.57,60), 'b--')
ax1.text (0,0.575, '0.57')

ax1.legend(['SINR by Max Power', 'SINR by Opt Power'], loc = 'upper left')

ax1.set_xlabel('SINR (dB)')
ax1.set_ylabel ('CDF')
#ax2.legend()
#ax1.legend()

ax1.grid()
plt.show()

'''

plt.figure()

plt.plot(np.sort(g0_list), np.linspace(0,1,len(g0_list)), label = 'SNR from power optimization')

def get_sinr(n=30, ratio=0.1):
    sinr_list =np.array([])
    snr_list = np.array([])
    for i in range(n):
        file = 'data_60/data_%d/bs_loc/bs_%d.csv'%(ii,i)
        d = pd.read_csv(file, delimiter = '\t')
        sinr = d['SINR']#[d['ue_type']==0]
        snr = d['SNR']#[d['ue_type']==0]
        snr_list = np.append(snr_list, snr)
        sinr_list = np.append(sinr_list,sinr)
    return sinr_list, snr_list
_, snr = get_sinr(30)
plt.plot(np.sort(snr), np.linspace(0,1,len(snr)), label = 'SNR from simulation')
plt.grid()
plt.legend()
'''