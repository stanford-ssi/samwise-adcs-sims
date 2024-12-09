import numpy as np

from simwise.math.quaternion import (
    dcm_to_quaternion, 
    quaternion_to_dcm, 
    rotate_vector_by_quaternion, 
    error_quaternion,
    regularize_quaternion
)
from simwise.math.dcm import passive_dcm

def test_dcm_to_quaternion_and_rotate_vector_by_quaternion():
    vec = np.array([1.0, 0.0, 0.0])
    
    # passive rotation of 90 degrees CCW about the z-axis
    dcm = passive_dcm("z", np.pi/2)

    q = dcm_to_quaternion(dcm)
    rotated_vec = rotate_vector_by_quaternion(vec, q)
    assert np.allclose(rotated_vec, dcm @ vec)

    # passive rotation of 45 degrees CW about the y-axis
    dcm = passive_dcm("y", np.pi/4)
    q = dcm_to_quaternion(dcm)
    rotated_vec = rotate_vector_by_quaternion(vec, q)
    assert np.allclose(rotated_vec, dcm @ vec)

    # passive rotation of -30 degrees about the x-axis
    vec = np.array([0.0, 1.0, 0.0])
    dcm = passive_dcm("x", -np.pi/6)
    q = dcm_to_quaternion(dcm)
    rotated_vec = rotate_vector_by_quaternion(vec, q)
    assert np.allclose(rotated_vec, dcm @ vec)

def test_quaternion_to_dcm():
    q = np.array([0.70710678, 0.0, 0.0, 0.70710678])
    dcm = quaternion_to_dcm(q)
    assert np.allclose(dcm, passive_dcm("z", -np.pi/2))
    assert np.allclose(q, dcm_to_quaternion(dcm))

    q = np.array([0.92387953, 0.0, 0.38268343, 0.0])
    dcm = quaternion_to_dcm(q)
    assert np.allclose(dcm, passive_dcm("y", np.pi/4))
    assert np.allclose(q, dcm_to_quaternion(dcm))

    q = np.array([0.8660254, 0.5, 0.0, 0.0])
    dcm = quaternion_to_dcm(q)
    assert np.allclose(dcm, passive_dcm("x", -np.pi/3))
    assert np.allclose(q, dcm_to_quaternion(dcm))

def test_error_quaternion():
    q = np.array([0.70710678, 0.0, 0.0, 0.70710678])
    q_hat = np.array([0.70710678, 0.0, 0.0, -0.70710678])
    error = error_quaternion(q, q_hat)
    error = regularize_quaternion(error)
    q = dcm_to_quaternion(passive_dcm("z", np.pi))
q = regularize_quaternion(q)
    assert np.allclose(error, dcm_to_quaternion(passive_dcm("z", -np.pi)))