"""
Runge-Kutta 4th order integrator

@ Author: Lundeen Cahilly
@ Date: 2026-02-20
"""

import numpy as np
from simwise.satellite.state import SatelliteState

def rk4(state: SatelliteState, dt: float, f: callable, sigma_u: float = 0.001) -> SatelliteState:
    k1 = f(state, state.t)

    s2 = state + (dt/2) * k1
    s2.q = s2.q.normalize()
    k2 = f(s2, state.t + dt/2)

    s3 = state + (dt/2) * k2
    s3.q = s3.q.normalize()
    k3 = f(s3, state.t + dt/2)

    s4 = state + dt * k3
    s4.q = s4.q.normalize()
    k4 = f(s4, state.t + dt)

    result = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    result.t = state.t + dt
    result.q = result.q.normalize()
    result.gyro_bias = state.gyro_bias + np.random.normal(0, sigma_u * np.sqrt(dt), 3)
    return result