"""
Sun model.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from simwise.satellite.data_structures.state import SatelliteState
from simwise.utils.time import mjd

def sun_vector_eci(state: SatelliteState):
    t = Time(mjd(state), format='mjd', scale='utc')
    r_sun_barycentric = get_body_barycentric('sun', t)
    r_earth_barycentric = get_body_barycentric('earth', t)
    r_sun_eci = r_sun_barycentric - r_earth_barycentric
    return r_sun_eci.xyz.to_value('m')  # [m] in ECI frame