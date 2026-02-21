"""
J2 perturbation

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from simwise.constants import R_EARTH, MU_EARTH, J2_EARTH

def j2_perturbation(state, params):
    x, y, z = state.r
    r = np.linalg.norm(state.r)
    r2 = r**2

    factor = -3 * MU_EARTH * J2_EARTH * R_EARTH**2 / (2 * r**5)

    ax = factor * x * (1 - 5 * z**2 / r2)
    ay = factor * y * (1 - 5 * z**2 / r2)
    az = factor * z * (3 - 5 * z**2 / r2)

    return np.array([ax, ay, az])  # [m/s^2]