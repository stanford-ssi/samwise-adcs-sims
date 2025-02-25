import pytest
import numpy as np
from simwise.data_structures.parameters import Parameters
from simwise.math.area_projection import create_rotation_matrix, define_satellite_vertices, project_prism, calculate_projected_area
from simwise.constants import *
from simwise.world_model.atmosphere import *
from simwise.forces.drag import dragPertubationTorque

@pytest.fixture
def params():
    """Create and return a Parameters object for testing."""
    return Parameters()

def test_basic_functionality(params):
    """Test basic functionality and output structure."""
    e_angles = np.array([0, 0, 0])
    velocity = np.array([7600, 0, 0])
    altitude = 400000  # 400 km
    
    torque = dragPertubationTorque(params, e_angles, velocity, altitude)
    
    # Should return results for 3 solar activity levels
    assert len(torque) == 3
    
    # Each torque should be a 3D vector
    for t in torque:
        assert isinstance(t, np.ndarray)
        assert t.shape == (3,)

def test_zero_velocity(params):
    """Test that zero velocity produces zero torque."""
    e_angles = np.array([0, 0, 0])
    velocity = np.array([0, 0, 0])
    altitude = 400000
    
    torque = dragPertubationTorque(params, e_angles, velocity, altitude)
    
    # With zero velocity, expect zero torque
    for t in torque:
        assert np.allclose(t, np.zeros(3), atol=1e-10)

def test_altitude_effect(params):
    """Test that torque decreases with increasing altitude."""
    e_angles = np.array([0, 0, 0])
    velocity = np.array([7600, 0, 0])
    
    low_altitude = 200000  # 200 km
    high_altitude = 800000  # 800 km
    
    torque_low = dragPertubationTorque(params, e_angles, velocity, low_altitude)
    torque_high = dragPertubationTorque(params, e_angles, velocity, high_altitude)
    
    # Torque should be greater at lower altitudes due to higher atmospheric density
    for i in range(3):
        assert np.linalg.norm(torque_low[i]) > np.linalg.norm(torque_high[i])

def test_velocity_magnitude_effect(params):
    """Test that torque increases with velocity magnitude."""
    e_angles = np.array([0, 0, 0])
    altitude = 400000  # 400 km
    
    low_velocity = np.array([1000, 0, 0])
    high_velocity = np.array([8000, 0, 0])
    
    torque_low = dragPertubationTorque(params, e_angles, low_velocity, altitude)
    torque_high = dragPertubationTorque(params, e_angles, high_velocity, altitude)
    
    # Torque should be greater with higher velocity
    for i in range(3):
        assert np.linalg.norm(torque_low[i]) < np.linalg.norm(torque_high[i])

def test_velocity_direction(params):
    """Test that torque direction changes with velocity direction."""
    e_angles = np.array([0, 0, 0])
    altitude = 400000
    
    velocity_x = np.array([7600, 0, 0])
    velocity_z = np.array([0, 0, 7600])  # Same magnitude, different direction
    
    torque_x = dragPertubationTorque(params, e_angles, velocity_x, altitude)
    torque_z = dragPertubationTorque(params, e_angles, velocity_z, altitude)
    
    # Different velocity directions should produce different torque vectors
    # Check at least one solar activity level has different torque direction
    different_direction = False
    for i in range(3):
        if not np.allclose(torque_x[i], torque_z[i], atol=1e-10):
            different_direction = True
            break
    assert different_direction

def test_orientation_effect(params):
    """Test that satellite orientation affects torque."""
    altitude = 400000
    velocity = np.array([7600, 0, 0])
    
    # Different orientations
    orientation1 = np.array([0, 0, 0])
    orientation2 = np.array([np.pi/2, 0, 0])
    
    torque1 = dragPertubationTorque(params, orientation1, velocity, altitude)
    torque2 = dragPertubationTorque(params, orientation2, velocity, altitude)
    
    # Different orientations should produce different torques
    different_torque = False
    for i in range(3):
        if not np.allclose(torque1[i], torque2[i], atol=1e-10):
            different_torque = True
            break
    assert different_torque

def test_solar_activity_levels(params):
    """Test that different solar activity levels produce different torques."""
    e_angles = np.array([0, 0, 0])
    velocity = np.array([7600, 0, 0])
    altitude = 30000