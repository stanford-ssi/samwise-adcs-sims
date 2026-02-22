"""
IMU simulator

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

from simwise.satellite.state import SatelliteState
from simwise.math.quaternion import Quaternion
from simwise.world.b_field import b_field_dipole

import numpy as np

def read_imu(state, params):
    w_eci = state.w
    q_eci2body = state.q
    w_body = q_eci2body.rot(w_eci)
    
    # add gyro bias
    w_body += state.gyro_bias

    # add gaussian noise
    noise = np.random.normal(0, 0.01, 3)
    state.gyro_bias += noise
    w_body += noise

    return w_body