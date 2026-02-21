"""
Runge-Kutta 4th order integrator

@ Author: Lundeen Cahilly
@ Date: 2026-02-20
"""

import numpy as np
from simwise.satellite.state import SatelliteState

def rk4(state: SatelliteState, dt: float, f: callable) -> SatelliteState:
    k1 = f(state, state.t)
    state.q = state.q.normalize()

    k2 = f(state + (dt/2) * k1, state.t + dt/2)
    state.q = state.q.normalize()

    k3 = f(state + (dt/2) * k2, state.t + dt/2)
    state.q = state.q.normalize()

    k4 = f(state + dt * k3, state.t + dt)
    state.q = state.q.normalize()

    result = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    result.t = state.t + dt
    return result