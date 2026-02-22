"""
Sun sensors.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from simwise.satellite.state import SatelliteState
from simwise.satellite.params import SatelliteParams
from simwise.world.sun import sun_eci

# TODO: add individual sensor readings
def read_sun_sensors(state, params):
    q_eci2body = state.q
    sun_eci = sun_eci(state)
    sun_body = q_eci2body.rot(sun_eci)

    # add gaussian noise
    noise = np.random.normal(0, 0.01, 3)
    sun_body += noise

    return sun_body