"""
spherical.py:  Methods for spherical coordinates
"""

import numpy as np




def cart_to_sph(d):
    """
    Cartesian to spherical coordinates.

    Parameters
    ----------
    d : (n,3) array
        vector of positions

    Returns:
    -------
    r:  (n,) array
        radius of each point
    phi, theta:  (n,) arrays
        azimuth and inclination angles in degrees
    """

    # Compute radius
    r = np.sqrt(np.sum(d ** 2, axis=1))
    r = np.maximum(r, 1e-8)

    # Compute angle of departure
    phi = np.arctan2(d[:, 1], d[:, 0]) * 180 / np.pi
    theta = np.arccos(d[:, 2] / r) * 180 / np.pi

    return r, phi, theta


def sph_to_cart(r, phi, theta):
    """
    Spherical coordinates to cartesian coordinates

    Parameters
    ----------
    r:  (n,) array
        radius of each point
    phi, theta:  (n,) arrays
        azimuth and inclination angles in degrees

    Returns
    -------
    d : (n,3) array
        vector of positions

    """

    # Convert to radians
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180

    # Convert to cartesian
    d0 = r * np.cos(phi) * np.sin(theta)
    d1 = r * np.sin(phi) * np.sin(theta)
    d2 = r * np.cos(theta)
    d = np.stack((d0, d1, d2), axis=-1)

    return d


def spherical_add_sub(phi0, theta0, phi1, theta1, sub:bool=True):
    """
    Angular addition and subtraction in spherical coordinates

    For addition, we start with a vector at (phi0,theta0), then rotate by
    theta1 in the (x1,x3) plance and then by phi1 in the (x1,x2) plane.
    For subtraction, we start with a vector at (phi0,theta0), then rotate by
    -phi1 in the (x1,x2) plane and then by -theta1 in the (x1,x3) plane.


    Parameters
    ----------
    phi0, theta0 : arrays of same size
        (azimuth,inclination) angle of the initial vector in degrees
    phi1, theta1 : arrays of same size
        (azimuth,inclination) angle of the rotation
    sub:  boolean
        if true, the angles are subtracted.  otherwise, they are added

    Returns
    -------
    phi2, theta2 : arrays of same size as input
        (azimuth,inclination) angle of the rotated vector

    """
    #  ^ z
    #  |
    #  |
    #  BS ----> x-axis ( this is the direction of antenna array)
    # Convert to radians
    theta0 = np.pi / 180 * theta0
    theta1 = np.pi / 180 * theta1
    phi0 = np.pi / 180 * phi0
    phi1 = np.pi / 180 * phi1

    if sub:
        # Find unit vector in direction of (theta0,phi0)
        x1 = np.sin(theta0) * np.cos(phi0) # theta is inclination angle
        x2 = np.sin(theta0) * np.sin(phi0)
        x3 = np.cos(theta0)

        y1 = x1 * np.cos(phi1) + x2 * np.sin(phi1)
        y2 = -x1 * np.sin(phi1) + x2 * np.cos(phi1)
        y3 = x3

        # Rotate by theta1 around y axis
        z1 = y1 * np.cos(theta1) - y3 * np.sin(theta1)
        z3 = y1 * np.sin(theta1) + y3 * np.cos(theta1)
        z2 = y2
        z1 = np.minimum(1, np.maximum(-1, z1))
        # Compute the angle of the transformed vector
        # we use the (z3,z2,z1) coordinate system
        phi2 = np.arctan2(z2, z3) * 180 / np.pi
        theta2 = np.arcsin(z1) * 180 / np.pi

    else:

        # Find unit vector in direction of (theta0,phi0)
        x3 = np.cos(theta0) * np.cos(phi0)
        x2 = np.cos(theta0) * np.sin(phi0)
        x1 = np.sin(theta0)

        # Rotate by theta1
        y1 = x1 * np.cos(theta1) + x3 * np.sin(theta1)
        y3 = -x1 * np.sin(theta1) + x3 * np.cos(theta1)
        y2 = x2

        # Rotate by phi1.
        z1 = y1 * np.cos(phi1) - y2 * np.sin(phi1)
        z2 = y1 * np.sin(phi1) + y2 * np.cos(phi1)
        z3 = y3
        z3 = np.minimum(1, np.maximum(-1, z3))

        # Compute angles
        phi2 = np.arctan2(z2, z1) * 180 / np.pi
        theta2 = np.arccos(z3) * 180 / np.pi
    return phi2, theta2

def rotation(phi1, theta1, x1, x2,x3):
    # counter-clock wise rotation
    # Rotate by phi1 around z axis

    y1 = x1 * np.cos(phi1) - x2 * np.sin(phi1)
    y2 = x1 * np.sin(phi1) + x2 * np.cos(phi1)
    y3 = x3

    # Rotate by theta1 around y axis
    z1 = y1 * np.cos(theta1) + y3 * np.sin(theta1)
    z3 = -y1 * np.sin(theta1) + y3 * np.cos(theta1)
    z2 = y2
    #z1 = np.minimum(1, np.maximum(-1, z1))
    return z1,z2,z3


def GCS_LCS_conversion(rot_angle: dict(), theta: list(), phi: list()):
    # GCS to LCS conversion function following 3GPP 38.901, 7.1-7, 7.1-8, and 7.1-15
    # set rotation angles of UE
    # alpha: bearing angle, rotation about z axis
    # beta: downtilt angle, rotation about y axis
    # gamma: slant angle, rotation about x axis

    # theta, phi are radians
    # alpha, beta, gamma are all radians
    # example of the use
    '''
     rot_angle = dict()
    rot_angle['alpha'] = np.deg2rad(30) # 30 degree horizontal rotation
    rot_angle['beta'] = np.deg2rad(180) # -90 degree down tilted
    rot_angle['gamma'] = 0
    theta = [np.deg2rad(97)]
    phi  = [np.deg2rad(20)]
    loc_angle, _ = GCS_LCS_conversion(rot_angle, theta, phi)

    loc_angle['theta_prime']  = np.rad2deg(loc_angle['theta_prime'])
    loc_angle['phi_prime']  = np.rad2deg(loc_angle['phi_prime'])
    '''
    alpha, beta, gamma = rot_angle['alpha'], rot_angle['beta'], rot_angle['gamma']

    k = np.cos(beta) * np.cos(gamma) * np.cos(theta) + (
                np.sin(beta) * np.cos(gamma) * np.cos(phi - alpha) - np.sin(gamma) * np.sin(phi - alpha)) * np.sin(theta)
    k = np.minimum(k, 1)
    theta_prime = np.arccos(k)  # 7.1-7

    a_jb = (np.cos(beta) * np.sin(theta) * np.cos(phi - alpha) - np.sin(beta) * np.cos(theta) \
            + 1j * (np.cos(beta) * np.sin(gamma) * np.cos(theta)
                    + (np.sin(beta) * np.sin(gamma) * np.cos(phi - alpha)
                       + np.cos(gamma) * np.sin(phi - alpha)) * np.sin(theta)))
    phi_prime = np.angle(a_jb)  # 7.1-8

    a_jb2 = (np.sin(gamma) * np.cos(theta) * np.sin(phi - alpha) + np.cos(gamma) *
             (np.cos(beta) * np.sin(theta) - np.sin(beta) * np.cos(theta) * np.cos(phi - alpha)) \
             + 1j * (np.sin(gamma) * np.cos(phi - alpha) + np.sin(beta) * np.cos(gamma) * np.sin(phi - alpha)))
    Psi = np.angle(a_jb2)  # 7.1-15

    # return local angles and rotation angle
    local_angle = {'theta_prime': theta_prime, 'phi_prime': phi_prime}
    # return angles are all radians
    return local_angle, Psi



