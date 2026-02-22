"""
Tests for the equations of motion.

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import unittest
import numpy as np
from simwise.math import Quaternion, rk4
from simwise.satellite import SatelliteState, SatelliteParams
from simwise.dynamics import state_dot
from simwise.constants import R_EARTH, MU_EARTH


I_body = np.array([[0.01861, 0.00529, 0.0001439],
                   [0.00529, 0.01833, 0.0000584709],
                   [0.0001439, 0.0000584709, 0.01558]])
m = 2.0

def run_sim(state, params, torques=[], perturbations=[], dt=0.1, tf=100.0):
    history = [state]
    f = lambda s, t: state_dot(s, params, torques=torques, perturbations=perturbations)
    while state.t < tf:
        state = rk4(state, dt, f)
        history.append(state)
    return history


class TestAngularMomentumConservation(unittest.TestCase):
    """With no external torque, |I @ w| should be conserved."""

    def setUp(self):
        params = SatelliteParams(I=I_body, m=m)
        state = SatelliteState(
            q_eci2body=Quaternion(0, 0, 0, 1),
            w_eci=np.array([0.1, 0.05, 0.2]),
            r_eci=np.array([R_EARTH, 0.0, 0.0]),
            v_eci=np.array([0.0, 7905.0, 0.0]),
            t=0.0,
        )
        self.history = run_sim(state, params)
        self.I = I_body

    def L_mag(self, state):
        return np.linalg.norm(self.I @ state.w)

    def test_angular_momentum_conserved(self):
        L0 = self.L_mag(self.history[0])
        for state in self.history:
            self.assertAlmostEqual(self.L_mag(state), L0, places=6)


class TestOrbitalAngularMomentumConservation(unittest.TestCase):
    """With no external force, |r × v| should be conserved (Keplerian orbit)."""

    def setUp(self):
        params = SatelliteParams(I=I_body, m=m)
        v_circ = np.sqrt(MU_EARTH / R_EARTH)
        state = SatelliteState(
            q_eci2body=Quaternion(0, 0, 0, 1),
            w_eci=np.zeros(3),
            r_eci=np.array([R_EARTH, 0.0, 0.0]),
            v_eci=np.array([0.0, v_circ, 0.0]),
            t=0.0,
        )
        self.history = run_sim(state, params, dt=1.0, tf=5000.0)

    def h_mag(self, state):
        return np.linalg.norm(np.cross(state.r, state.v))

    def test_orbital_angular_momentum_conserved(self):
        h0 = self.h_mag(self.history[0])
        for state in self.history:
            self.assertAlmostEqual(self.h_mag(state) / h0, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
