"""
Tests for the orbital elements utilities.

@ Author: Lundeen Cahilly
@ Date: 2026-02-27
"""

import unittest
from simwise.utils.orbital_elements import state2coe, coe2state
import numpy as np

class TestOrbitalElements(unittest.TestCase):
    def test_roundtrip_coe2state2coe(self):
        """coe -> state -> coe should recover original elements."""
        from simwise.satellite.data_structures.state import SatelliteState
        from simwise.math.quaternion import Quaternion

        state0 = SatelliteState(Quaternion(0, 0, 0, 1), np.zeros(3), np.zeros(3), np.zeros(3))

        test_cases = [
            # SSO-like: a [m], e, i [deg], w [deg], W [deg], nu [deg]
            [6939.13e3, 0.0,   97.64, 0.0,  180.0, 0.0],
            # ISS-like
            [6778.0e3,  0.001, 51.6,  90.0,  45.0, 30.0],
            # Elliptical
            [8000.0e3,  0.2,   28.5,  60.0, 120.0, 45.0],
        ]

        for coe in test_cases:
            with self.subTest(coe=coe):
                state = coe2state(state0, coe)
                recovered = state2coe(state)
                np.testing.assert_allclose(recovered, coe, atol=1e-6,
                    err_msg=f"Round-trip failed for coe={coe}")

    def test_state2coe(self):
        # test
        pass