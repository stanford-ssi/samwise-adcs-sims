"""
Gravity gradient torque.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from simwise.constants import R_EARTH, MU_EARTH

def gravity_gradient(state, params):
    I = params.I
    r = state.r 
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag

    return np.cross(3 * MU_EARTH / r_mag**3 * r_hat, I @ r_hat) # [Nm]