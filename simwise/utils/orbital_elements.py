"""
Orbital elements utilities.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from simwise.satellite.data_structures.state import SatelliteState
from simwise.math import R1, R2, R3
from simwise.constants import MU_EARTH

# Cartesian state to classical orbital elements (COEs)
def state2coe(state):
    r = state.r
    v = state.v
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag
    v_mag = np.linalg.norm(v)

    # 1. specific angular momentum
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)

    # 2. inclination
    i = np.arctan2(np.sqrt(h_vec[0]**2 + h_vec[1]**2), h_vec[2])

    # 3. longitude of ascending node
    W = np.arctan2(h_vec[0], -h_vec[1])

    # 4. semi-major axis
    a = (2 / r_mag - v_mag**2 / MU_EARTH)**(-1)

    # 5. semi-latus rectum
    p = h**2 / MU_EARTH

    # 6. eccentricity
    e = np.sqrt(1 - p / a)

    # 7. argument of latitude
    K_hat = np.array([0, 0, 1])
    n = np.cross(K_hat, h_vec) # line of nodes
    n_mag = np.linalg.norm(n)
    u = np.arctan2(np.dot(np.cross(n, r), h_vec) / (n_mag * r_mag * h), np.dot(n, r) / (n_mag * r_mag))

    # 8. true anomaly
    n = np.sqrt(MU_EARTH / a**3)
    E = np.arctan2(np.dot(r, v) / (a**2 * n), 1 - r_mag / a)
    nu = np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)

    # 9. argument of periapsis
    w = u - nu

    # convert to degrees
    i = np.degrees(i)
    W = np.degrees(W)
    w = np.degrees(w)
    nu = np.degrees(nu)

    # wrap to 0 to 360 (even if negative)
    i = (i + 360) % 360
    W = (W + 360) % 360
    w = (w + 360) % 360
    nu = (nu + 360) % 360

    return np.array([a, e, i, W, w, nu])

# Classical orbital elements (COEs) to Cartesian state
def coe2state(state, coe):
    a, e, i, W, w, nu = coe

    # convert to radians
    i = np.radians(i)
    W = np.radians(W)
    w = np.radians(w)
    nu = np.radians(nu)

    # 1. semi-latus rectum (directly relates to h)
    p = a * (1 - e**2)

    # 2. position in perifocal frame
    r_mag = p / (1 + e * np.cos(nu))
    r_pqw = np.array([
        r_mag * np.cos(nu),
        r_mag * np.sin(nu),
        0
    ])

    # 3. velocity in perifocal frame
    v_pqw = np.array([
        -np.sqrt(MU_EARTH / p) * np.sin(nu),
        np.sqrt(MU_EARTH / p) * (e + np.cos(nu)),
        0
    ])

    # 4. transform to ECI frame
    r_eci = R3(-W) @ R1(-i) @ R3(-w) @ r_pqw
    v_eci = R3(-W) @ R1(-i) @ R3(-w) @ v_pqw

    return SatelliteState(state.q, state.w, r_eci, v_eci, state.t, state.mjd_epoch)

# # State to modified equinoctial elements (MEES) TODO: test!!
# def state2mee(state):
#     r = state.r
#     v = state.v
#     r_mag = np.linalg.norm(r)
#     r_hat = r / r_mag
#     v_mag = np.linalg.norm(v)

#     # 1. specific angular momentum
#     h_vec = np.cross(r, v)
#     h_mag = np.linalg.norm(h_vec)

#     # 2. semi-latus rectum
#     p = h_mag**2 / MU_EARTH

#     # 3. radial dot product
#     rdv = np.dot(r, v)

#     # 4. transverse unit vector
#     v_hat = (r_mag * v - rdv * r_hat) / h_mag

#     # 5. orientation elements
#     h = (h_vec[0] / h_mag) / (1 + h_vec[3] / h_mag)
#     k = (-h_vec[1] / h_mag) / (1 + h_vec[3] / h_mag)

#     # 6. auxiliary scalars
#     k2 = k**2 # :)
#     h2 = h**2
#     s2 = 1 + h2 + k2
#     tkh = 2 * k * h

#     # 7. eccentricity vector
#     e_vec = np.cross(v, h_vec) / MU_EARTH - r_hat

#     # 8. frame vectors
#     f_hat = 1 / s2 * np.array([1 - k2 + h2, tkh, -2 * k])
#     g_hat = 1 / s2 * np.array([tkh, 1 + k2 - h2, 2 * h])

#     # 9. eccentricity projections
#     f = np.dot(e_vec, f_hat)
#     g = np.dot(e_vec, g_hat)

#     # 10. true longitude
#     L = arctan2(r_hat[1] - v_hat[0], r_hat[0] - v_hat[1])
    
#     # convert to degrees
#     L = np.degrees(L)
#     L = (L + 360) % 360

#     return np.array([p, f, g, h, k, L])

# # Modified equinoctial elements (MEES) to state
# def mee2state(state, mee):
#     p, f, g, h, k, L = mee

#     # convert to radians
#     L = np.radians(L)

#     # 1. semi-latus rectum (directly relates to h)
#     return NONE # TODO: implement!

    