"""
Sun sensors.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from simwise.satellite.state import SatelliteState
from simwise.satellite.params import SatelliteParams


def read_sun_sensors(state, params):
    return np.zeros(3)