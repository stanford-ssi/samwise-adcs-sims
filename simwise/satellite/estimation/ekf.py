"""
Extended Kalman Filter for satellite estimation.

@ Author: Lundeen Cahilly
@ Date: 2026-02-27
"""

import numpy as np
from simwise.satellite.state import SatelliteState
from simwise.satellite.params import SatelliteParams
from simwise.utils.orbital_elements import state2coe, coe2state

class EKF:
    pass