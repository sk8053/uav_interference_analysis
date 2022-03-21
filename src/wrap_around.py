
import numpy as np

def wrap_around(dist_vec, network):
    dist_x, dist_y, dist_z = dist_vec.T
    ind_x = np.where(network.X_MAX - np.abs(dist_x) <= np.abs(dist_x))
    ind_y = np.where(network.Y_MAX - np.abs(dist_y) <= np.abs(dist_y))
    dist_x[ind_x] = (-1) * np.sign(dist_x[ind_x]) * (network.X_MAX - np.abs(dist_x[ind_x]))
    dist_y[ind_y] = (-1) * np.sign(dist_y[ind_y]) * (network.Y_MAX - np.abs(dist_y[ind_y]))
    return np.column_stack((dist_x, dist_y, dist_z))