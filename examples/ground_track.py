"""
Ground track simulation using simwise.

@ Author: Lundeen Cahilly
@ Date: 2026-02-26
"""

import numpy as np
from simwise import Quaternion, SatelliteState, SatelliteParams, Satellite, j2, date2mjd, coe2state, propagate
from simwise.constants import R_EARTH
from simwise.utils import build_groundtrack_fig, build_ecef_fig

# Orbital elements at epoch
# [m] and [deg]
a = 6939.13e3
e = 0.0
i = 97.64
W = 180.0
w = 0.0    # singular
nu = 0.0   # singular

I = np.eye(3)
m = 1.0  # [kg]

params = SatelliteParams(I=I, m=m)
state0 = SatelliteState(
    q_eci2body=Quaternion(0, 0, 0, 1),
    w_eci=np.zeros(3),
    r_eci=np.zeros(3),  # overwritten by coe2state
    v_eci=np.zeros(3),  # overwritten by coe2state
    t=0.0,
    mjd_epoch=date2mjd(2026, 2, 20),
)
state0 = coe2state(state0, [a, e, i, w, W, nu])

satellite = Satellite(state=state0, params=params)

propagate(
    satellite,
    torques=[],
    perturbations=[j2],
    dt=5,
    orbit_every=1,
    tf=86400.0*4,  # 4 days
)
build_groundtrack_fig(satellite.history).show()
build_ecef_fig(satellite.history).show()
