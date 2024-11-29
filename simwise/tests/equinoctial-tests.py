import numpy as np
from simwise.constants import *
from simwise.orbit.equinoctial import *

# TEST CASES:
def assert_close(calculated, expected, tolerance=0.1):
    """Simple function to check if values are close enough"""
    if abs(calculated - expected) > tolerance:
        raise AssertionError(f"Values not close enough: calculated={calculated}, expected={expected}")

def assert_vector_close(calculated, expected, tolerance=0.1):
    """Check if vector components are close enough"""
    if not np.allclose(calculated, expected, atol=tolerance):
        raise AssertionError(f"Vectors not close enough:\ncalculated={calculated}\nexpected={expected}")

def test_circular_polar_orbit():
    """Test altitude and velocity for a circular polar orbit"""
    # Constants
    radius_earth = 6371  # km
    mu_earth = 398600.4418  # km³/s²
    
    # Set up a circular polar orbit at 500 km altitude
    altitude = 500  # km
    a = radius_earth + altitude
    
    # Classical orbital elements for circular polar orbit
    # [a, e, i, Ω, ω, ν]
    coe = np.array([
        a,              # semi-major axis (km)
        0.0,           # eccentricity (circular)
        90.0,          # inclination (degrees) - polar orbit
        45.0,          # RAAN (degrees) - arbitrary for this test
        0.0,           # argument of perigee (undefined for circular)
        0.0            # true anomaly (arbitrary for circular)
    ])
    
    # Convert to state vectors
    rv = coe_to_rv(coe, mu_earth)
    
    # Test altitude
    calculated_altitude = get_altitude(rv)
    expected_altitude = altitude
    assert_close(calculated_altitude, expected_altitude)
    print(f"Polar orbit altitude test passed: {calculated_altitude:.1f} km")
    
    # Test velocity
    calculated_velocity = get_velocity(rv)
    expected_velocity = np.sqrt(mu_earth / a)  # circular orbit velocity
    assert_close(calculated_velocity, expected_velocity)
    print(f"Polar orbit velocity test passed: {calculated_velocity:.1f} km/s")

def test_elliptical_equatorial_orbit():
    """Test altitude and velocity for an elliptical equatorial orbit"""
    # Constants
    radius_earth = 6371  # km
    mu_earth = 398600.4418  # km³/s²
    
    # Perigee and apogee altitudes
    perigee_alt = 300  # km
    apogee_alt = 35793  # km (approximately GEO altitude)
    
    rp = radius_earth + perigee_alt
    ra = radius_earth + apogee_alt
    a = (rp + ra) / 2
    e = (ra - rp) / (ra + rp)
    
    # Classical orbital elements
    coe = np.array([
        a,              # semi-major axis (km)
        e,             # eccentricity
        0.0,           # inclination (degrees) - equatorial
        0.0,           # RAAN (degrees)
        0.0,           # argument of perigee
        0.0            # true anomaly (starting at perigee)
    ])
    
    # Convert to state vectors
    rv = coe_to_rv(coe, mu_earth)
    
    # Test altitude at perigee
    calculated_altitude = get_altitude(rv)
    expected_altitude = perigee_alt
    assert_close(calculated_altitude, expected_altitude)
    print(f"Equatorial orbit altitude test passed: {calculated_altitude:.1f} km")
    
    # Test velocity at perigee
    calculated_velocity = get_velocity(rv)
    expected_velocity = np.sqrt(mu_earth * (2/rp - 1/a))  # vis-viva equation
    assert_close(calculated_velocity, expected_velocity)
    print(f"Equatorial orbit velocity test passed: {calculated_velocity:.1f} km/s")


def test_molniya_orbit():
    """Test altitude and velocity for a Molniya-type orbit"""
    # Constants
    radius_earth = 6371  # km
    mu_earth = 398600.4418  # km³/s²
    
    # Molniya orbit characteristics
    perigee_alt = 1000  # km
    apogee_alt = 39830  # km
    
    rp = radius_earth + perigee_alt
    ra = radius_earth + apogee_alt
    a = (rp + ra) / 2
    e = (ra - rp) / (ra + rp)
    
    # Classical orbital elements
    coe = np.array([
        a,              # semi-major axis (km)
        e,             # eccentricity
        63.4,          # inclination (degrees) - critical inclination
        45.0,          # RAAN (degrees)
        270.0,         # argument of perigee
        0.0            # true anomaly (starting at perigee)
    ])
    
    # Convert to state vectors
    rv = coe_to_rv(coe, mu_earth)
    
    # Split into position and velocity components
    r = rv[:3]
    v = rv[3:]
    
    # Test overall altitude
    calculated_altitude = get_altitude(rv)
    expected_altitude = perigee_alt
    assert_close(calculated_altitude, expected_altitude)
    print(f"Molniya orbit altitude test passed: {calculated_altitude:.1f} km")
    
    # Test position components
    print("\nPosition components [x, y, z]:")
    print(f"  {r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f} km")
    
    # At perigee with argument of perigee = 270°, the position should be:
    # - primarily in -z direction (due to argument of perigee)
    # - tilted by inclination angle
    # - magnitude should be rp
    expected_magnitude = rp
    calculated_magnitude = np.linalg.norm(r)
    assert_close(calculated_magnitude, expected_magnitude)
    print(f"Position magnitude test passed: {calculated_magnitude:.1f} km")
    
    # Test velocity components
    print("\nVelocity components [vx, vy, vz]:")
    print(f"  {v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f} km/s")
    
    # Test velocity magnitude at perigee
    calculated_velocity = np.linalg.norm(v)
    expected_velocity = np.sqrt(mu_earth * (2/rp - 1/a))  # vis-viva equation
    assert_close(calculated_velocity, expected_velocity)
    print(f"Velocity magnitude test passed: {calculated_velocity:.1f} km/s")
    

if __name__ == '__main__':
    print("\nTesting circular polar orbit:")
    test_circular_polar_orbit()
    
    print("\nTesting elliptical equatorial orbit:")
    test_elliptical_equatorial_orbit()
    
    print("\nTesting Molniya orbit:")
    test_molniya_orbit()
    
    print("\nAll tests passed!")