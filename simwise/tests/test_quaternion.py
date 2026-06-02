"""
Tests for the quaternion class

@ Author: Lundeen Cahilly
@ Date: 2026-02-20
"""

import unittest
from simwise.math.quaternion import Quaternion
import numpy as np


class TestQuaternion(unittest.TestCase):
    def test_init(self):
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q.x, 1)
        self.assertEqual(q.y, 2)
        self.assertEqual(q.z, 3)
        self.assertEqual(q.w, 4)

    def test_add(self):
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        result = q1 + q2
        self.assertEqual(result.x, 6)
        self.assertEqual(result.y, 8)
        self.assertEqual(result.z, 10)
        self.assertEqual(result.w, 12)

    def test_mul_identity(self):
        q = Quaternion(1, 2, 3, 4)
        identity = Quaternion(0, 0, 0, 1)
        result = q * identity
        self.assertEqual(result.x, q.x)
        self.assertEqual(result.y, q.y)
        self.assertEqual(result.z, q.z)
        self.assertEqual(result.w, q.w)
    
    def test_rot(self):
        q = Quaternion.from_angle_axis(np.pi/2, np.array([1, 0, 0]))
        v = np.array([0, 1, 0])
        v_expected = np.array([0, 0, -1]) # passive
        result = q.rot(v)
        self.assertAlmostEqual(result[0], v_expected[0])
        self.assertAlmostEqual(result[1], v_expected[1])
        self.assertAlmostEqual(result[2], v_expected[2])
    
    def test_rot_active(self):
        q = Quaternion.from_angle_axis(np.pi/2, np.array([1, 0, 0]))
        v = np.array([0, 1, 0])
        v_expected = np.array([0, 0, 1]) # active
        result = q.rot_active(v)
        self.assertAlmostEqual(result[0], v_expected[0])
        self.assertAlmostEqual(result[1], v_expected[1])
        self.assertAlmostEqual(result[2], v_expected[2])


if __name__ == "__main__":
    unittest.main()