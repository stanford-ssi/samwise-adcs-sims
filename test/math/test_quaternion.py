import pytest
import numpy as np
from simwise.math.quaternion import *

@pytest.fixture
def test_quaternions():
    """Common test quaternions"""
    return {
        'identity': np.array([1.0, 0.0, 0.0, 0.0]),
        'x90': np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0]),  # 90° around x
        'y90': np.array([np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0]),  # 90° around y
        'z90': np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)]),  # 90° around z
        'arbitrary': np.array([0.5, 0.5, 0.5, 0.5]) / np.sqrt(1.0)  # Arbitrary normalized
    }

@pytest.fixture
def test_vectors():
    """Common test vectors"""
    return {
        '0': np.array([0, 0, 0]),
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
        'arbitrary': np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    }

def test_normalize_quaternion():
    """Test quaternion normalization"""
    # Test un-normalized quaternion
    q = np.array([2.0, 3.0, 4.0, 5.0])
    q_norm = normalize_quaternion(q)
    np.testing.assert_allclose(np.linalg.norm(q_norm), 1.0, rtol=1e-10)
    
    # Test already normalized quaternion
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_norm = normalize_quaternion(q)
    np.testing.assert_allclose(q_norm, q, rtol=1e-10)

def test_quaternion_multiply(test_quaternions):
    """Test quaternion multiplication"""
    # Identity multiplication
    q = test_quaternions['arbitrary']
    result = quaternion_multiply(test_quaternions['identity'], q)
    np.testing.assert_allclose(result, q, rtol=1e-10)
    
    # 90° rotations composition
    # Two 90° rotations around x should equal 180° rotation
    result = quaternion_multiply(
        test_quaternions['x90'], 
        test_quaternions['x90']
    )
    expected = np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0.0, 0.0])
    
    # Compare absolute values since the signs might differ
    np.testing.assert_allclose(np.abs(result), np.abs(expected), rtol=1e-10, atol=1e-15)

def test_quaternion_inverse(test_quaternions):
    """Test quaternion inverse"""
    for name, q in test_quaternions.items():
        # q * q^-1 should equal identity
        q_inv = quaternion_inverse(q)
        result = quaternion_multiply(q, q_inv)
        np.testing.assert_allclose(result, test_quaternions['identity'], rtol=1e-10)

def test_euler_quaternion_roundtrip():
    """Test euler to quaternion and back"""
    test_angles = [
        [0, 0, 0],  # Zero rotation
        [np.pi/2, 0, 0],  # 90° around x
        [0, np.pi/2, 0],  # 90° around y
        [0, 0, np.pi/2],  # 90° around z
        [np.pi/4, np.pi/3, np.pi/6]  # Arbitrary rotation
    ]
    
    for angles in test_angles:
        angles = np.array(angles)
        # Convert to quaternion and back
        q = euler_to_quaternion(angles)
        euler_back = quaternion_to_euler(q)
        
        # Handle the case where angles wrap around
        angles_wrapped = np.mod(angles + np.pi, 2*np.pi) - np.pi
        euler_back_wrapped = np.mod(euler_back + np.pi, 2*np.pi) - np.pi
        
        np.testing.assert_allclose(angles_wrapped, euler_back_wrapped, rtol=1e-10)

def test_rotate_vector(test_quaternions, test_vectors):
    """Test vector rotation by quaternion"""
    # 90° rotation around z should take x to y
    rotated = rotate_vector_by_quaternion(
        test_vectors['x'], 
        test_quaternions['z90']
    )
    # Add absolute tolerance for near-zero comparisons
    np.testing.assert_allclose(rotated, test_vectors['y'], rtol=1e-10, atol=1e-15)
    
    # Any rotation of zero vector should be zero
    zero = np.zeros(3)
    for q in test_quaternions.values():
        rotated = rotate_vector_by_quaternion(zero, q)
        np.testing.assert_allclose(rotated, zero, rtol=1e-10, atol=1e-15)
        
    # Test that magnitude is preserved
    v = test_vectors['arbitrary']
    for q in test_quaternions.values():
        rotated = rotate_vector_by_quaternion(v, q)
        np.testing.assert_allclose(np.linalg.norm(rotated), np.linalg.norm(v), rtol=1e-10)

def test_quaternions_to_axis_angle(test_quaternions):
    """Test quaternion to axis-angle conversion"""
    # Identity rotation should give zero angle
    theta, axis = quaternions_to_axis_angle(
        test_quaternions['identity'],
        test_quaternions['identity']
    )
    assert np.abs(theta) < 1e-10
    np.testing.assert_allclose(axis, np.zeros(3), atol=1e-15)
    
    # 90° x rotation
    theta, axis = quaternions_to_axis_angle(
        test_quaternions['identity'],
        test_quaternions['x90']
    )
    np.testing.assert_allclose(np.abs(theta), np.pi/2, rtol=1e-10)
    np.testing.assert_allclose(np.abs(axis), [1, 0, 0], rtol=1e-10, atol=1e-15)

def test_dcm_to_quaternion():
    """Test DCM to quaternion conversion"""
    # Identity matrix should give identity quaternion
    dcm = np.eye(3)
    q = dcm_to_quaternion(dcm)
    np.testing.assert_allclose(q, np.array([1, 0, 0, 0]), rtol=1e-10, atol=1e-15)
    
    # 90° rotation around z
    c, s = np.cos(np.pi/2), np.sin(np.pi/2)
    dcm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    q = dcm_to_quaternion(dcm)
    expected = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    np.testing.assert_allclose(np.abs(q), np.abs(expected), rtol=1e-10, atol=1e-15)

def test_special_cases():
    """Test special and edge cases"""
    # Test near-zero angles
    small_angle = np.array([1e-7, 1e-7, 1e-7])
    q = euler_to_quaternion(small_angle)
    angles_back = quaternion_to_euler(q)
    np.testing.assert_allclose(small_angle, angles_back, rtol=1e-5, atol=1e-15)
    
    # Test angles near pi
    near_pi = np.array([np.pi-1e-7, 0, 0])
    q = euler_to_quaternion(near_pi)
    angles_back = quaternion_to_euler(q)
    # Allow for wrapping
    diff = np.min([np.abs(angles_back - near_pi), np.abs(angles_back + near_pi)], axis=0)
    assert np.all(diff < 1e-5)