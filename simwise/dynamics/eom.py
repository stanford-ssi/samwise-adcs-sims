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

def w_dot(state, params, torques=[]):
    w = state.w
    I = params.I
    tau = sum((t(state, params) for t in torques), np.zeros(3))
    return np.linalg.inv(I) @ (tau - np.cross(w, I @ w))

def r_dot(state):
    v = state.v
    return v

def v_dot(state, params, perturbations=[]):
    r = state.r
    r_mag = np.linalg.norm(r)
    a = sum((p(state, params) for p in perturbations), np.zeros(3))
    return -MU_EARTH * r / r_mag**3 + a

def attitude_dot(state, params, torques=[]):
    return SatelliteState(q_dot(state), w_dot(state, params, torques), np.zeros(3), np.zeros(3))

def orbit_dot(state, params, perturbations=[]):
    return SatelliteState(Quaternion(0, 0, 0, 0), np.zeros(3), r_dot(state), v_dot(state, params, perturbations))

def state_dot(state, params, torques=[], perturbations=[]):
    return SatelliteState(q_dot(state), w_dot(state, params, torques), r_dot(state), v_dot(state, params, perturbations))
