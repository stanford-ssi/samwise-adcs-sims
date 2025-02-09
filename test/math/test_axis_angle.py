import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from simwise.math.axis_angle import vectors_to_axis_angle


def test_vectors_to_axis_angle():
    # Test cases
    test_cases = [
        # Parallel vectors (0 degrees)
        (np.array([1, 0, 0]), np.array([1, 0, 0]), 0),
        
        # 90 degree rotation cases
        (np.array([1, 0, 0]), np.array([0, 1, 0]), np.pi/2),
        (np.array([0, 1, 0]), np.array([0, 0, 1]), np.pi/2),
        
        # 180 degree rotation
        (np.array([1, 0, 0]), np.array([-1, 0, 0]), np.pi),
        
        # 45 degree rotation
        (np.array([1, 0, 0]), np.array([1, 1, 0])/np.sqrt(2), np.pi/4),
    ]
    
    for v1, v2, expected_angle in test_cases:
        axis, angle = vectors_to_axis_angle(v1, v2)
        
        # Check angle
        assert np.abs(angle - expected_angle) < 1e-10
        
        # For non-zero angles, verify the axis is normalized
        if angle > 0:
            assert np.abs(np.linalg.norm(axis) - 1.0) < 1e-10
            
        # Verify rotation works by applying it
        rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)
        rotated_v1 = rotation_matrix @ v1
        assert_array_almost_equal(rotated_v1, v2, decimal=10)

def rotation_matrix_from_axis_angle(axis, angle):
    """Helper function to convert axis-angle to rotation matrix"""
    if angle == 0:
        return np.eye(3)
        
    k = axis
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)