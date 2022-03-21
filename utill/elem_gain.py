import numpy as np
import matplotlib.pyplot as plt

G_max = 8
G_s = -22
phi_hpbw = 65
theta = 102.8
phi_list = np.linspace(-np.pi, np.pi, 1000)
phi_list = np.rad2deg(phi_list)

G_phi_list = []
for phi in phi_list:
    if np.abs(phi) < theta:
        G_phi = G_max *10**(-0.3*(2*phi/phi_hpbw)**2)
    else:
        G_phi = G_s
    G_phi_list.append(G_phi)
plt.plot(phi_list, G_phi_list)
plt.grid()
plt.show()