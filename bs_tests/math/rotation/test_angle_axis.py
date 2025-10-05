import numpy as np

import simwise.math.rotation.angle_axis as AA
import simwise.math.unit_conversions as conversion

def test_x_and_y_axes():
    i_hat = np.array([1,0,0])
    j_hat = np.array([0,1,0])

    angle, axis = AA.axis_angle_between_vectors(i_hat, j_hat)
    assert angle == np.pi/2
    assert all(np.isclose(
        axis, np.array([0, 0, 1])
    ))

    angle, axis = AA.axis_angle_between_vectors(i_hat, j_hat, degrees=True)
    assert angle == 90
    assert all(np.isclose(
        axis, np.array([0, 0, 1])
    ))

def test_small_angle():
    i_hat = np.array([1,0,0])
    rot_angle = 1e-3 * conversion.DEG_TO_RAD
    i_hat_plus_z = np.array([np.cos(rot_angle), 0, np.sin(rot_angle)])

    angle, axis = AA.axis_angle_between_vectors(i_hat, i_hat_plus_z, degrees=True)
    assert np.isclose(angle, 1e-3)
    assert all(np.isclose(
        axis, np.array([0, -1, 0])
    ))

def test_zero():
    i_hat = np.array([1,0,0])

    angle, _ = AA.axis_angle_between_vectors(i_hat, i_hat)
    assert angle == 0
    angle, _ = AA.axis_angle_between_vectors(i_hat, i_hat, degrees=True)
    assert angle == 0