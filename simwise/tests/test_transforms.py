"""
Tests for the quaternion class

@ Author: Lundeen Cahilly
@ Date: 2026-02-20
"""

import unittest
from simwise.utils.transforms import blh2ecef, ecef2blh, eci2ecef, ecef2eci, enu2ecef, ecef2enu, enu2azel
import numpy as np


class TestTransforms(unittest.TestCase):
    def test_blh2ecef(self):
        # Durand Building, Stanford, CA
        B = np.radians(37.42687087)
        L = np.radians(-122.17333307)
        H = 48.0
        r_ecef = blh2ecef(np.array([B, L, H]))

        # checked against https://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
        r_ecef_expected = np.array([-2700.42e3, -4292.624e3, 3855.151e3])
        assert np.allclose(r_ecef, r_ecef_expected, atol=1e-2)

    def test_ecef2blh(self):
        r_ecef = np.array([2700e3, -4000e3, 4300e3])
        blh = ecef2blh(r_ecef)

        # checked against https://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
        blh_expected = np.array([np.radians(41.88984), np.radians(304.01935), 95098])
        assert np.allclose(blh, blh_expected, atol=1e-2)

    

if __name__ == "__main__":
    unittest.main()