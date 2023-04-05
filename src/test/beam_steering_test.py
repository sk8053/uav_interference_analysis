## author: Sundeep rangan
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
path = os.path.abspath('..')
if not path in sys.path:
    sys.path.append(path)
import mmwchanmod as mmc

from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray
import mmwchanmod.sim.antenna
elem = Elem3GPP(thetabw=65, phibw=80)
plt.figure()
elem.plot_pattern(nphi=360, plot_type='rect_phi')
plt.ylabel('Gain (dBi)')
plt.grid()
plt.savefig('fig/3gpp_gain.png')

plt.figure()
theta = np.array([0,45])
elem.plot_pattern(nphi=360, theta=theta, plot_type='polar_phi')
leg_str = []
for t in theta:
    leg_str.append( 'theta=%d' % t)
plt.legend(leg_str, loc='center left')
plt.savefig('fig/ant_polar.png')

# Antenna element
elem = Elem3GPP(thetabw=90, phibw=80)

# Number of antenna elements in the y and z direction respectively
nant = np.array([8,8])

fc=28e9 # carrier frequency for narrowband response

# Array
arr = URA(elem=elem, nant=nant, fc=fc)
plt.figure()
# Plot the antenna positions in the y and z plane
plt.plot(arr.elem_pos[:,1], arr.elem_pos[:,2], 'o')
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.grid()
plt.savefig('fig/ant_config.png')

theta0 = [0, 0, 40]
phi0 = [0, 50, 50]
nplot = len(theta0)

fig, ax = plt.subplots(1, nplot, figsize=(10, 5))
for i in range(nplot):
    # Get the beamform vector
    w = arr.conj_bf(phi0[i], theta0[i])

    # Plot the antenna pattern
    # Note the use of passing vmin and vmax, arguments to the plot
    # This makes sure all plots are on the same colorscale
    phi, theta, v, axi, im = \
        arr.plot_pattern(w=w, nphi=360, ntheta=180, \
                         plot_type='2d', vmin=-20, vmax=30, ax=ax[i])
    axi.grid()
    axi.plot(phi0[i], theta0[i], 'ro', ms=5)

    # Compute peak gain
    peak_gain = np.max(v)
    axi.set_title('peak gain=%4.1f dBi' % peak_gain)

# Add a colorbar
plt.tight_layout()
fig.subplots_adjust(bottom=0.1, right=0.93, top=0.9)
cax = plt.axes([0.95, 0.1, 0.02, 0.8])
_ = fig.colorbar(im, cax=cax)
plt.savefig('fig/beam_pattern_by_different_bf_vectors.png')

# Desired angle of maximum gain
phi0 = 60
theta0 = -45

# Create the rotated array
arr_rot = RotatedArray(arr,phi0=phi0,theta0=theta0)

# Get BF vector in direction of maximum gain
w = arr_rot.conj_bf(phi0, theta0)

plt.figure()
# Get the antenna pattern
phi, theta, v, axi, im =\
   arr_rot.plot_pattern(w=w,nphi=360, ntheta = 180,\
                plot_type='2d', vmin=-20, vmax=30)
axi.grid()
axi.plot(phi0, theta0, 'ro', ms=5)
plt.colorbar(im)
plt.savefig('fig/beam_pattern_of_rotated_array.png')

plt.show()