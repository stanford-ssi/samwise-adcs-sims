"""
Coordinate transformations

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from simwise.math.quaternion import Quaternion
from simwise.constants import R_EARTH, E_WGS84
from simwise.math.rot import R1, R2, R3

def blh2ecef(B, L, H):
    N = R_EARTH / np.sqrt(1 - E_WGS84**2 * np.sin(B)**2)
    x = (N + H) * np.cos(B) * np.cos(L)
    y = (N + H) * np.cos(B) * np.sin(L)
    z = (N * (1 - E_WGS84**2) + H) * np.sin(B)
    return np.array([x, y, z])

def ecef2blh(r_ecef):
    # iteratively solve for B
    for i in range(10):
        N = R_EARTH / np.sqrt(1 - E_WGS84**2 * np.sin(B)**2)
        B = np.arctan2(r_ecef[2], np.sqrt(r_ecef[0]**2 + r_ecef[1]**2))
        H = np.linalg.norm(r_ecef) - N
    return np.array([B, L, H])

def eci2ecef(r_eci, gmst):
    R_eci2ecef = R3(gmst)
    return R_eci2ecef @ r_eci

def ecef2eci(r_ecef, gmst):
    R_ecef2eci = R3(-gmst)
    return R_ecef2eci @ r_ecef

def ecef2enu(r_ecef, B, L, H):
    r_obs = blh2ecef(B, L, H)
    R_ecef2enu = R_ecef2enu(B, L)
    r_enu = R_ecef2enu @ (r_ecef - r_obs)
    return r_enu

def enu2ecef(r_enu, B, L, H):
    r_obs = blh2ecef(B, L, H)
    R_enu2ecef = R3(-(90 + L)) @ R1(90 - B) @ r_enu
    r_ecef = R_enu2ecef @ r_enu + r_obs
    return r_ecef

def R_ecef2enu(B, L):
    return R1(90 - B) @ R3(90 + L)

def R_enu2ecef(B, L):
    return R3(-(90 + L)) @ R1(B - 90)

def enu2azel(r_enu):
    E = r_enu[0]
    N = r_enu[1]
    U = r_enu[2]
    az = np.arctan2(N, E)
    el = np.arctan2(U, np.sqrt(E**2 + N**2))
    return az, el