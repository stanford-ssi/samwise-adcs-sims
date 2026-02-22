"""
Full attitude propagator using simwise.

@ Author: Lundeen Cahilly
@ Date: 2026-02-20
"""

import numpy as np
from simwise import Quaternion, SatelliteState, SatelliteParams, gravity_gradient, j2, date2mjd, propagate
from simwise.constants import R_EARTH 
from simwise.utils.plots import build_attitude_fig, build_orbit_fig

I_body = np.array([[0.01861, 0.00529, 0.0001439],
              [0.00529, 0.01833, 0.0000584709],
              [0.0001439, 0.0000584709, 0.01558]])  # [kg m^2]
m = 2.0 # [kg]

params = SatelliteParams(I=I_body, m=m)
state0 = SatelliteState(
    q_eci2body=Quaternion(0, 0, 0, 1),
    w_eci=np.array([0.0, 0.0, 0.1]),
    r_eci=np.array([R_EARTH + 350e3, 0.0, 0.0]),
    v_eci=np.array([-200.0, 5445.48, 5545.48]),
    t=0.0,
    mjd_epoch=date2mjd(2026, 2, 20),
)

history = propagate(
    state0, params,
    torques=[gravity_gradient],
    perturbations=[j2],
    dt=0.1,
    orbit_every=100,
    tf=1.5 * 3600.0,
)
build_attitude_fig(history).show()
build_orbit_fig(history).show()
