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
data_dir = 'data_60_new/data_%d'%ii

# Parameters
tx_pow = 23
nf = 6
bw = 400e6
EkT = -174
ntrials = 30
ndrops = 7

rate_list = [np.zeros(0), np.zeros(0)]
g0_list =[]
for it in range(ntrials):

    # Select a random drop
    idrop = np.random.randint(0, ndrops, 1)

    # Print status
    print('trial = %d, drop=%d' % (it, idrop))

    # Read data
    pl_path = os.path.join(data_dir, 'pl_eff_mat/pl_eff_matrix_%d.csv' % idrop)
    df_pl = pd.read_csv(pl_path, sep=' ')
    nue, nbs = df_pl.shape
    pl = np.array(df_pl)

    # Find the index of the serving BS for each UE from the
    # lowest path loss
    Iserv = np.argmin(pl, axis=1)

    # For each base station, select one UE.  This creates a set of
    # links, where in each link i:  ue_ind[i] is the index of the UE
    # and bs_ind[i] is the base station
    ue_ind = []
    bs_ind = []
    for ibs in range(nbs):
        # Find UEs served by the BS
        I = np.where(Iserv == ibs)[0]
        n = len(I)

        # If there is at least one UE served by the BS,
        # select a random UE and add it
        if n > 0:
            i = np.random.randint(n)
            iue = I[i]
            bs_ind.append(ibs)
            ue_ind.append(iue)

    ue_ind = np.array(ue_ind)
    bs_ind = np.array(bs_ind)

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
        rate = util_max.util_eval(p, \
                                  return_snr_rate=True)[1]
        rate_list[i] = np.hstack((rate_list[i], rate))

# Plot the CDF of the rates over all the links
for rate in rate_list:
    rate = np.sort(rate)
    n = len(rate)
    plt.semilogx(rate, np.arange(n) / n)
np.savetxt('rate_%d.txt'%ii, rate_list)
plt.grid()
plt.xlabel('Spec efficiency (bps/Hz)')
plt.ylabel('CDF')
plt.legend(['Max Power', 'Opt Power'])

plt.savefig('rate_cdf.png')

plt.figure()

plt.plot(np.sort(g0_list), np.linspace(0,1,len(g0_list)))

def get_sinr(n=30, ratio=0.1):
    sinr_list =np.array([])
    snr_list = np.array([])
    for i in range(n):
        file = 'data_60_new/data_50/bs_loc/bs_%d.csv'%(i)
        d = pd.read_csv(file, delimiter = '\t')
        sinr = d['SINR'][d['ue_type']==0]
        snr = d['SNR'][d['ue_type']==0]
        snr_list = np.append(snr_list, snr)
        sinr_list = np.append(sinr_list,sinr)
    return sinr_list, snr_list
_, snr = get_sinr(9)
plt.plot(np.sort(snr), np.linspace(0,1,len(snr)))

plt.show()
