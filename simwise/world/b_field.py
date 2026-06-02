"""
Magnetic field model.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np

MU_EARTH = np.array([0.0, 0.0, 8.0e22]) # [A m^2]

# use simple dipole model
# B = (3 * (r dot m) * r - r^2 * m) / r^5
def b_earth_dipole(state):
    r_eci = state.r
    r_mag = np.linalg.norm(r_eci)   
    r_hat = r_eci / r_mag
    b_eci = (3 * np.dot(r_hat, MU_EARTH) * r_hat - r_mag**2 * MU_EARTH) / r_mag**5
    return b_eci # [T]