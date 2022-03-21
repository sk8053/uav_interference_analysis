import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
bw = 400e6
def plot_rate(rate, L = 'full power'):
    plt.plot(np.sort(rate), np.linspace(0,1,len(rate)), label = L)
rate_50= np.loadtxt('../rate_50.txt')
rate_10 = np.loadtxt('../rate_10.txt')
rate_25 = np.loadtxt('../rate_25.txt')
rate_5 = np.loadtxt('../rate_5.txt')
rate_5 = rate_5.T *bw/1e6
rate_10 = rate_10.T *bw/1e6
rate_50 = rate_50.T *bw/1e6
rate_25 = rate_25.T *bw/1e6

#plot_rate(rate_5[:,0], L = 'full power')
plot_rate(rate_5[:,0], L = 'optimal power with 5:95')
plot_rate(rate_10[:,0], L = 'optimal power with 10:90')

plot_rate(rate_25[:,0], L = 'optimal power with 25:75')
plot_rate(rate_50[:,0], L = 'optimal power with 50:50')

#plot_rate(rate_5[:,1], L = 'full power with 5:95')
#plot_rate(rate_10[:,1], L = 'full power with 10:80')

#plot_rate(rate_25[:,1], L = 'full power with 25:75')
#plot_rate(rate_50[:,1], L = 'full power with 50:50')


plt.xlabel('Throughput (Mbps)')
plt.ylabel('CDF')
plt.legend()
plt.grid()
plt.show()
