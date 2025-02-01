import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from simwise.constants import MU_EARTH
from simwise.math.quaternion import quaternion_multiply, rotate_vector_by_quaternion
from simwise.forces.gravity_gradient import gravity_gradient_perturbation_torque
from simwise.data_structures.parameters import Parameters

@pytest.fixture
def sample_orbit_radius():
    """Sample orbital radius at 450km altitude"""
    R_EARTH = 6371000  # meters
    return R_EARTH + 450000  # 450km orbit

@pytest.fixture
def inertia_cases():
    """Various inertia tensor test cases including actual satellite inertia"""
    params = Parameters()
    return {
        'zero': np.array([0.0, 0.0, 0.0]),
        'equal': np.array([1.0, 1.0, 1.0]),
        'rod': np.array([1.0, 10.0, 10.0]),  # Long axis along x
        'disk': np.array([10.0, 10.0, 1.0]),  # Disk in xy plane
        'realistic': np.array([14.21e-9, 40.87e-9, 32.02e-9]),  # From example
        'satellite': params.inertia  # Actual satellite inertia
    }

def test_zero_torque_all_inertias(inertia_cases):
    """Test zero torque condition for all inertia cases when aligned with radius vector"""
    r_vector = np.array([1.0, 0.0, 0.0])  # Along x-axis
    q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    for case_name, inertia in inertia_cases.items():
        if case_name == 'equal':  # Should be exactly zero for equal inertias
            torque = gravity_gradient_perturbation_torque(q, r_vector, inertia)
            assert_array_almost_equal(torque, np.zeros(3), 
                                    err_msg=f"Failed zero torque test for {case_name} case")
        else:  # For non-equal inertias, torque should be small but may not be exactly zero
            torque = gravity_gradient_perturbation_torque(q, r_vector, inertia)
            assert np.linalg.norm(torque) < 1e-10, f"Torque too large for aligned {case_name} case"

@pytest.mark.parametrize("inertia_key", ['rod', 'disk', 'satellite'])
def test_maximum_torque_45_degrees(inertia_cases, inertia_key):
    """Test maximum torque at 45 degree rotation for different inertia configurations"""
    r_vector = np.array([1.0, 0.0, 0.0])
    inertia = inertia_cases[inertia_key]
    
    # Choose rotation axis based on inertia configuration
    if inertia_key == 'disk':
        # For disk, rotate about y-axis to tilt disk out of plane
        q = np.array([np.cos(np.pi/8), 0.0, np.sin(np.pi/8), 0.0])
    else:
        # For rod and satellite, keep original z-axis rotation
        q = np.array([np.cos(np.pi/8), 0.0, 0.0, np.sin(np.pi/8)])
    
    torque = gravity_gradient_perturbation_torque(q, r_vector, inertia)
    
    # Calculate theoretical maximum
    r_mag = np.linalg.norm(r_vector)
    max_theoretical = (3 * MU_EARTH / (2 * r_mag**3)) * max(
        abs(inertia[0] - inertia[1]),
        abs(inertia[1] - inertia[2]),
        abs(inertia[2] - inertia[0])
    )
    
    # For 45° rotation, torque should be near maximum
    torque_magnitude = np.linalg.norm(torque)
    assert_allclose(torque_magnitude, max_theoretical, rtol=1e-2, 
                    err_msg=f"Maximum torque test failed for {inertia_key} case")

@pytest.mark.parametrize("inertia_key", ['realistic', 'satellite'])
def test_torque_scaling_with_distance_realistic(sample_orbit_radius, inertia_cases, inertia_key):
    """Test 1/r³ scaling with realistic inertias and orbital distances"""
    r_vector1 = np.array([sample_orbit_radius, 0.0, 0.0])
    r_vector2 = np.array([2*sample_orbit_radius, 0.0, 0.0])
    
    inertia = inertia_cases[inertia_key]
    q = np.array([np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0])  # 45° about y
    
    torque1 = gravity_gradient_perturbation_torque(q, r_vector1, inertia)
    torque2 = gravity_gradient_perturbation_torque(q, r_vector2, inertia)
    
    # Torque should scale with 1/r³
    expected_ratio = 1/8  # (r1/r2)³ = (1/2)³ = 1/8
    actual_ratio = np.linalg.norm(torque2) / np.linalg.norm(torque1)
    
    assert_allclose(actual_ratio, expected_ratio, rtol=1e-10,
                    err_msg=f"Distance scaling test failed for {inertia_key} case")

def test_torque_scaling_with_inertia_satellite(inertia_cases):
    """Test torque scaling with actual satellite inertia"""
    r_vector = np.array([1.0, 0.0, 0.0])
    q = np.array([np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0])  # 45° about y
    
    satellite_inertia = inertia_cases['satellite']
    scaled_inertia = 2 * satellite_inertia
    
    torque1 = gravity_gradient_perturbation_torque(q, r_vector, satellite_inertia)
    torque2 = gravity_gradient_perturbation_torque(q, r_vector, scaled_inertia)
    
    assert_allclose(torque2, 2*torque1, rtol=1e-10,
                    err_msg="Satellite inertia scaling test failed")

@pytest.mark.parametrize("inertia_key", ['realistic', 'satellite'])
def test_90_degree_rotation_realistic(inertia_cases, inertia_key):
    """Test 90 degree rotation behavior with realistic inertias"""
    r_vector = np.array([1.0, 0.0, 0.0])
    inertia = inertia_cases[inertia_key]
    
    # 90 degree rotation about z-axis
    q = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
    
    torque = gravity_gradient_perturbation_torque(q, r_vector, inertia)
    
    # Verify expected torque direction for 90° rotation
    # Primary torque should be about z-axis
    assert abs(torque[2]) > abs(torque[0]), f"Unexpected torque direction for {inertia_key} 90° rotation"
    assert abs(torque[2]) > abs(torque[1]), f"Unexpected torque direction for {inertia_key} 90° rotation"

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_satellite_principal_axes(inertia_cases, axis):
    """Test torque behavior when satellite is aligned with principal axes"""
    inertia = inertia_cases['satellite']
    
    # Create position vector along each principal axis
    r_vector = np.zeros(3)
    r_vector[axis] = 1.0
    
    q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    torque = gravity_gradient_perturbation_torque(q, r_vector, inertia)
    assert_array_almost_equal(torque, np.zeros(3), 
                            err_msg=f"Non-zero torque when aligned with principal axis {axis}")