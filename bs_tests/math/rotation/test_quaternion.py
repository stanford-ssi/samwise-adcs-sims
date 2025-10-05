import numpy as np

import simwise.math.rotation.quaternion as Q
import simwise.math.unit_conversions as conversion

# --------------------------------------------------
#   Basic
# --------------------------------------------------

class TestConjugate:
    def test_quat_conjugate(self):
        # With list
        result = Q.conj([1,0,0,0])
        expected = np.array([1,0,0,0])
        assert np.array_equal(result,expected)

        result = Q.conj([1,1,1,1])
        expected = np.array([1,-1,-1,-1])
        assert np.array_equal(result,expected)

        # With numpy array
        q = np.array([1,0,0,0])
        result = Q.conj(q)
        assert np.array_equal(q,result)

        q = np.array([1,1,1,1])
        result = Q.conj(q)
        expected = np.array([1,-1,-1,-1])
        assert np.array_equal(result,expected)

class TestInverse:
    def test_inverse(self):
        """Relies on a working multiplication function for this to work"""
        # Defined as 
        #   q_inv * q = 1
        #   q * q_inv = 1
        q = np.array([1,2,3,4])
        q_inv = Q.inv(q)

        assert all(np.isclose(
            Q.multiply(q, q_inv), np.array([1,0,0,0])
        ))
        assert all(np.isclose(
            Q.multiply(q_inv, q), np.array([1,0,0,0])
        ))

class TestUnit:
    def test_unit(self):
        result = Q.unit([1,0,0,0])
        assert np.array_equal(
            result, np.array([1,0,0,0])
        )

        result = Q.unit([1,1,1,1])
        assert np.array_equal(
            result, np.array([0.5, 0.5, 0.5, 0.5])
        )

    def test_tolerance(self):
        # Less than tolerance
        result = Q.unit([.01, .01, .01, .01], tol=0.1) # tolerance: 0.02 < 0.1
        assert np.array_equal(
            result, np.zeros(4)
        )

        # Equal to tolerence
        result = Q.unit([.01, .01, .01, .01], tol=0.02) # tolerance: 0.02 == 0.02
        assert np.array_equal(
            result, np.array([.5, .5, .5, .5])
        )

        # Greater than tolerance (same as equal)
        result = Q.unit([.01, .01, .01, .01], tol=0.0001) # tolerance: 0.02 > 0.0001
        assert np.array_equal(
            result, np.array([.5, .5, .5, .5])
        )

# --------------------------------------------------
#   Angles
# --------------------------------------------------

class TestAxisAngleToQ:

    @staticmethod
    def angle_axis_to_q_unit_test(angle_rad, axis, q_expected):
        result = Q.angle_axis_to_q(angle_rad, axis, False)

        assert all(np.isclose(
            result, q_expected
        ))

    def test_identity(self):
        self.angle_axis_to_q_unit_test(0, [1,0,0], [1, 0, 0, 0])
        self.angle_axis_to_q_unit_test(0, [1,1,0], [1, 0, 0, 0])
        self.angle_axis_to_q_unit_test(0, [1,0,1], [1, 0, 0, 0])

    def test_180_deg(self):
        self.angle_axis_to_q_unit_test(180 * conversion.DEG_TO_RAD, [1,0,0], [0, 1, 0, 0])
        self.angle_axis_to_q_unit_test(180 * conversion.DEG_TO_RAD, [0,1,0], [0, 0, 1, 0])
        self.angle_axis_to_q_unit_test(180 * conversion.DEG_TO_RAD, [0,0,1], [0, 0, 0, 1])

    def test_90_deg(self):
        self.angle_axis_to_q_unit_test(90 * conversion.DEG_TO_RAD, [1,0,0], [1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        self.angle_axis_to_q_unit_test(90 * conversion.DEG_TO_RAD, [0,1,0], [1/np.sqrt(2), 0, 1/np.sqrt(2), 0])
        self.angle_axis_to_q_unit_test(90 * conversion.DEG_TO_RAD, [0,0,1], [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

    def test_all_axes(self):
        self.angle_axis_to_q_unit_test(90 * conversion.DEG_TO_RAD, [1,1,1], [1/np.sqrt(2), 1/np.sqrt(2*3), 1/np.sqrt(2*3), 1/np.sqrt(2*3)])
        self.angle_axis_to_q_unit_test(90 * conversion.DEG_TO_RAD, [-1,-1,-1], [1/np.sqrt(2), -1/np.sqrt(2*3), -1/np.sqrt(2*3), -1/np.sqrt(2*3)])

    def test_example_in_pdf(self):
        # https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf
        self.angle_axis_to_q_unit_test(2 * np.pi / 3, [1,1,1], [.5, .5, .5, .5])

    def test_with_degrees(self):
        result = Q.angle_axis_to_q(90, [1,1,1], True)

        assert all(np.isclose(
            result, np.array([1/np.sqrt(2), 1/np.sqrt(2*3), 1/np.sqrt(2*3), 1/np.sqrt(2*3)])
        ))

class TestQToAxisAngle:
    
    @staticmethod
    def q_to_axis_angle_unit_test(q, expected_angle_rad, expected_axis):
        angle, axis = Q.q_to_axis_angle(q, False)

        assert np.isclose(expected_angle_rad, angle)
        assert all(np.isclose(
            expected_axis, axis
        ))

    def test_identity(self):
        self.q_to_axis_angle_unit_test([1, 0, 0, 0], 0, [0,0,0])

    def test_180_deg(self):
        self.q_to_axis_angle_unit_test([0, 1, 0, 0], 180 * conversion.DEG_TO_RAD, [1,0,0])
        self.q_to_axis_angle_unit_test([0, 0, 1, 0], 180 * conversion.DEG_TO_RAD, [0,1,0])
        self.q_to_axis_angle_unit_test([0, 0, 0, 1], 180 * conversion.DEG_TO_RAD, [0,0,1])

    def test_90_deg(self):
        self.q_to_axis_angle_unit_test([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], 90 * conversion.DEG_TO_RAD, [1,0,0])
        self.q_to_axis_angle_unit_test([1/np.sqrt(2), 0, 1/np.sqrt(2), 0], 90 * conversion.DEG_TO_RAD, [0,1,0])
        self.q_to_axis_angle_unit_test([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], 90 * conversion.DEG_TO_RAD, [0,0,1])

    def test_all_axes(self):
        self.q_to_axis_angle_unit_test([1/np.sqrt(2), 1/np.sqrt(2*3), 1/np.sqrt(2*3), 1/np.sqrt(2*3)],
                                       90 * conversion.DEG_TO_RAD,
                                       [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
        self.q_to_axis_angle_unit_test([1/np.sqrt(2), -1/np.sqrt(2*3), -1/np.sqrt(2*3), -1/np.sqrt(2*3)],
                                       90 * conversion.DEG_TO_RAD,
                                       [-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)])

    def test_example_in_pdf(self):
        # https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf
        self.q_to_axis_angle_unit_test([.5, .5, .5, .5], 2 * np.pi / 3, [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])

    def test_with_degrees(self):
        angle, axis = Q.q_to_axis_angle([1/np.sqrt(2), 1/np.sqrt(2*3), 1/np.sqrt(2*3), 1/np.sqrt(2*3)], True)
        expected_axis = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
        assert np.isclose(90, angle)
        assert all(np.isclose(
            expected_axis, axis
        ))

# --------------------------------------------------
#   Algebra
# --------------------------------------------------

class TestMultiply:
    def test_pdf_example(self):
        q1 = np.array([3, 1, -2, 1])
        q2 = np.array([2, -1, 2, 3])
        result = Q.multiply(q1, q2)
        expected = np.array([8, -9, -2, 11])

        assert all(np.isclose(
            result, expected
        ))

class TestApply:

    def test_z_around_x_axis_active(self):
        """
        Starts with quaternion representing a 90deg rotation around the x axis.
        The matrix rotates the vector `V` `[0,0,1]` to produce `V'` `[0,1,0]`
        """

        angle = 90 * conversion.DEG_TO_RAD
        rotation_axis = np.array([1, 0, 0])
        
        quat = Q.angle_axis_to_q(angle, rotation_axis)

        vector = np.array([0,0,1])
        result = Q.apply(quat, vector, rotation_type="active")

        expected = np.array([0, -1, 0])

        assert all(np.isclose(
            result, expected
        ))

    def test_z_around_x_axis_passive(self):
        """
        Starts with quaternion representing a 90deg rotation around the x axis.
        The matrix expresses the vector in coordiante system A (`V_a`= `[0,0,1]`) 
        as the same vector from a rotated coordinate system B (`V_b` = `[0,1,0]`)
        """

        angle = 90
        rotation_axis = np.array([1, 0, 0])
        
        quat = Q.angle_axis_to_q(angle, rotation_axis, degrees=True)

        vector = np.array([0,0,1])
        result = Q.apply(quat, vector, rotation_type="passive")

        expected = np.array([0, 1, 0])

        assert all(np.isclose(
            result, expected
        ))

    def test_small_x_rotation_active(self):
        """
        Starts with quaternion representing a 90deg rotation around the x axis.
        The matrix rotates the vector `V` `[0,0,1]` to produce `V'` `[0,-sin(1e-6 deg),cos(1e-6 deg)]`
        """

        angle = 1e-6
        rotation_axis = np.array([1, 0, 0])
        
        quat = Q.angle_axis_to_q(angle, rotation_axis, degrees=True)

        vector = np.array([0,0,1])
        result = Q.apply(quat, vector, rotation_type="active")

        expected = np.array([0, -np.sin(angle * conversion.DEG_TO_RAD), np.cos(angle * conversion.DEG_TO_RAD)])
        assert all(np.isclose(
            result, expected
        ))

    def test_small_x_rotation_passive(self):
        """
        Starts with quaternion representing a 90deg rotation around the x axis.
        The matrix rotates the vector `V` `[0,0,1]` to produce `V'` `[0,sin(1e-6 deg),cos(1e-6 deg)]`
        """

        angle = 1e-6
        rotation_axis = np.array([1, 0, 0])
        
        quat = Q.angle_axis_to_q(angle, rotation_axis, degrees=True)

        vector = np.array([0,0,1])
        result = Q.apply(quat, vector, rotation_type="passive")

        expected = np.array([0, np.sin(angle * conversion.DEG_TO_RAD), np.cos(angle * conversion.DEG_TO_RAD)])
        assert all(np.isclose(
            result, expected
        ))

    def test_weird_rotations_active(self):

        # All axes
        angle = 120
        rotation_axis = np.array([1, 1, 1])
        
        quat = Q.angle_axis_to_q(angle, rotation_axis, degrees=True)

        vector = np.array([0,0,1])
        result = Q.apply(quat, vector, rotation_type="active")

        expected = np.array([1, 0, 0])
        assert all(np.isclose(
            result, expected
        ))

        # Rotating that specific vector
        angle = 90
        rotation_axis = np.array([0, 0, 1])
        
        quat = Q.angle_axis_to_q(angle, rotation_axis, degrees=True)

        vector = np.array([1,1,1])
        result = Q.apply(quat, vector, rotation_type="active")

        expected = np.array([-1, 1, 1])
        assert all(np.isclose(
            result, expected
        ))