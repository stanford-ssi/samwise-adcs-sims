from simwise.satellite.state import SatelliteState
from simwise.math.quaternion import Quaternion

import numpy as np

def read_magnetometer(state, params):
    b_eci = np.array([0.0, 0.0, 0.0])
    q_eci2body = state.q
    b_body = q_eci2body.rot(b_eci)
    return b_body