"""
Magnetometer sensor.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

from simwise.satellite.state import SatelliteState
from simwise.math.quaternion import Quaternion
from simwise.world.b_field import b_field_dipole

import numpy as np

def read_magnetometer(state, params):
    r_eci = state.r
    b_eci = b_field_dipole(r_eci)
    q_eci2body = state.q
    b_body = q_eci2body.rot(b_eci)
    return b_body