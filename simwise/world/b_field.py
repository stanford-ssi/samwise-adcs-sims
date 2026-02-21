"""
Magnetic field model.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np

# use simple dipole model for now
# B = (3 * (r dot m) * r - r^2 * m) / r^5
def b_field_dipole(r_eci, m):
    r_mag = np.linalg.norm(r_eci)
    r_hat = r_eci / r_mag
    b_eci = (3 * (r_hat @ m) * r_hat - r_mag**2 * m) / r_mag**5
    return b_eci # [T]