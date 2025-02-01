import pytest
import numpy as np
from datetime import datetime
import math
from simwise.world_model.magnetic_field import magnetic_field
from simwise.constants import EARTH_RADIUS_M

# Constants for testing
STANDARD_JD = 2460000.0  # Example Julian Date
TEST_TOLERANCE = 1e-6
EXPECTED_FIELD_RANGE = (20000, 70000)  # nT

# Helper functions
def calculate_field_magnitude(field_components):
    """Calculate magnitude of magnetic field vector"""
    if field_components is None:
        return None
    return np.sqrt(sum(x*x for x in field_components))

def is_within_expected_range(magnitude):
    """Check if field magnitude is within expected range"""
    if magnitude is None:
        return False
    return EXPECTED_FIELD_RANGE[0] <= magnitude <= EXPECTED_FIELD_RANGE[1]

# Fixtures
@pytest.fixture
def standard_date():
    return STANDARD_JD

# Geographic boundary test cases
@pytest.mark.parametrize("lla_wgs84, expected_valid", [
    ((0, 0, 0), True),  # Null Island
    ((45, 90, 1000), True),  # Mid-latitude
    ((89.9, 0, 0), True),  # North pole
    ((-89.9, 0, 0), True),  # South pole
    ((0, 180, 0), True),  # Date line
    ((0, -180, 0), True),  # Date line opposite
    ((91, 0, 0), False),  # Invalid latitude >90
    ((-91, 0, 0), False),  # Invalid latitude <-90
    ((0, 181, 0), False),  # Invalid longitude >180
    ((0, -181, 0), False),  # Invalid longitude <-180
])
def test_geographic_boundaries(lla_wgs84, expected_valid, standard_date):
    if expected_valid:
        result = magnetic_field(lla_wgs84, standard_date)
        assert result is not None, f"Expected valid result for {lla_wgs84}"
        assert len(result) == 3, f"Expected 3 components, got {len(result)}"
        magnitude = calculate_field_magnitude(result)
        assert magnitude is not None, f"Failed to calculate magnitude for {result}"
        assert is_within_expected_range(magnitude), \
            f"Field magnitude {magnitude} not within expected range {EXPECTED_FIELD_RANGE} for location {lla_wgs84}"
    else:
        with pytest.raises(ValueError):
            magnetic_field(lla_wgs84, standard_date)

# Altitude test cases
@pytest.mark.parametrize("altitude", [
    0,  # Sea level
    1000,  # 1km
    10000,  # 10km
    100000,  # 100km
    -1000,  # Below sea level
])
def test_altitude_scaling(altitude, standard_date):
    base_location = (0, 0, 0)
    test_location = (0, 0, altitude)
    
    base_field = magnetic_field(base_location, standard_date)
    test_field = magnetic_field(test_location, standard_date)
    
    assert base_field is not None
    assert test_field is not None
    
    base_magnitude = calculate_field_magnitude(base_field)
    test_magnitude = calculate_field_magnitude(test_field)
    
    # Field should decrease with altitude following roughly 1/r³ law
    if altitude > 0:
        expected_ratio = ((EARTH_RADIUS_M*10e3) / (EARTH_RADIUS_M*10e3 + altitude))**3
        actual_ratio = test_magnitude / base_magnitude
        assert abs(actual_ratio - expected_ratio) < 0.1  # 10% tolerance

# # Type testing
# @pytest.mark.parametrize("invalid_input", [
#     (("0", "0", "0"), STANDARD_JD),  # Strings
#     ((None, None, None), STANDARD_JD),  # None values
#     (([], [], []), STANDARD_JD),  # Empty lists
#     (None, STANDARD_JD),  # Missing lla_wgs84
# ])
# def test_invalid_types(invalid_input):
#     with pytest.raises(Exception):
#         magnetic_field(*invalid_input)

# Special locations test
@pytest.mark.parametrize("location, expected_properties", [
    ((80.7, -72.7, 0), {"vertical_dominant": True}),  # Magnetic north
    ((-64.3, 136.2, 0), {"vertical_dominant": True}),  # Magnetic south
    ((0, -174, 0), {"horizontal_dominant": True}),  # Magnetic equator
])
def test_special_locations(location, expected_properties, standard_date):
    result = magnetic_field(location, standard_date)
    assert result is not None
    
    Be, Bn, Bd = result
    horizontal_magnitude = np.sqrt(Be*Be + Bn*Bn)
    
    if expected_properties.get("vertical_dominant"):
        assert abs(Bd) > horizontal_magnitude
    
    if expected_properties.get("horizontal_dominant"):
        assert horizontal_magnitude > abs(Bd)

# Output format testing
def test_output_format(standard_date):
    result = magnetic_field((0, 0, 0), standard_date)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(x, float) for x in result)

# Performance testing
@pytest.mark.performance
def test_performance():
    start_time = datetime.now()
    for lat in range(-80, 81, 20):
        for lon in range(-160, 161, 40):
            magnetic_field((lat, lon, 0), STANDARD_JD)
    execution_time = (datetime.now() - start_time).total_seconds()
    assert execution_time < 1.0  # Should complete within 1 second

# Temporal variation testing
@pytest.mark.parametrize("test_date", [
    2440000.0,  # Past
    2450000.0,  # Mid
    2460000.0,  # Recent
])
def test_temporal_variation(test_date):
    location = (0, 0, 0)
    result = magnetic_field(location, test_date)
    assert result is not None
    magnitude = calculate_field_magnitude(result)
    assert is_within_expected_range(magnitude)

# Field continuity testing
def test_field_continuity(standard_date):
    # Test continuity across longitude boundary
    result1 = magnetic_field((0, 179.9, 0), standard_date)
    result2 = magnetic_field((0, -179.9, 0), standard_date)
    
    assert result1 is not None
    assert result2 is not None
    
    # Field should be nearly continuous across the date line
    diff = np.array(result1) - np.array(result2)
    assert np.all(np.abs(diff) < 1000)  # Less than 1000nT difference

if __name__ == "__main__":
    pytest.main([__file__])