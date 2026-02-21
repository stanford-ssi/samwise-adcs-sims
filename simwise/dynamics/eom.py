"""
Equations of motion for satellite dynamics

@ Author: Lundeen Cahilly
@ Date: 2026-02-20
"""

import numpy as np
from simwise.math.quaternion import Quaternion
from simwise.satellite.state import SatelliteState
from simwise.constants import MU_EARTH

def q_dot(state):
    q = state.q
    w = state.w
    return 0.5 * Quaternion(w[0], w[1], w[2], 0) * q

def w_dot(state, params, tau):
    w = state.w
    return np.linalg.inv(params.I) @ (tau - np.cross(w, params.I @ w))

def r_dot(state):
    r = state.r
    v = state.v
    return v

def v_dot(state, params, F):
    v = state.v
    r = np.linalg.norm(state.r)
    return -MU_EARTH * state.r / r**3 + F / params.m # treat F as perturbation

def attitude_dot(state, params, tau):
    return SatelliteState(q_dot(state), w_dot(state, params, tau), state.r, state.v)

def orbit_dot(state, params, F):
    return SatelliteState(state.q, state.w, r_dot(state), v_dot(state, params, F))

def state_dot(state, params, tau, F):
    return attitude_dot(state, params, tau) + orbit_dot(state, params, F)
