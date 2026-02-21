"""
Coordinate transformations

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from simwise.math.quaternion import Quaternion
from simwise.constants import R_EARTH, E_WGS84
from simwise.utils.rot import R1, R2, R3

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