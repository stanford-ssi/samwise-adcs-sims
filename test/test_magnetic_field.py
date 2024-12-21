import pytest
from simwise.world_model.magnetic_field import magnetic_field

# Test function
# USE ECF
def test_magnetic_field():
    # try:
    #     result = magnetic_field(90, 0, 100, 2460663.201412037)
    #     if result is not None:
    #         print(f"Magnetic field at 90°N, 0°E, 100km altitude:")
    #         print(f"North component = {result['north'].item():.1f} nT")
    #         print(f"East component = {result['east'].item():.1f} nT")
    #         print(f"Down component = {result['down'].item():.1f} nT")
    #         print(f"Total field = {result['total'].item():.1f} nT")
    # except Exception as e:
    #     print(f"Failed to calculate magnetic field: {str(e)}")
        # Input parameters
    latitude = 90      # Degrees
    longitude = 0      # Degrees
    altitude = 100     # Kilometers
    jd = 2460663.201412037  # Julian Date

    try:
        # Call the function
        result = magnetic_field([latitude, longitude, altitude], jd)

        # Assertions to verify results
        assert result is not None, "Result is None"
        assert "north" in result, "Missing 'north' component"
        assert "east" in result, "Missing 'east' component"
        assert "down" in result, "Missing 'down' component"
        assert "total" in result, "Missing 'total' component"

        # Validate the range of values (customize these ranges as needed)
        assert -1e5 < result['north'].item() < 1e5, "'north' component out of range"
        assert -1e5 < result['east'].item() < 1e5, "'east' component out of range"
        assert -1e5 < result['down'].item() < 1e5, "'down' component out of range"
        assert result['total'].item() > 0, "'total' field is non-positive"

    except Exception as e:
        pytest.fail(f"Failed to calculate magnetic field: {str(e)}")