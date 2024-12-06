from simwise.world_model.atmosphere import low_solar_activity, moderate_solar_activity, high_solar_activity

def test_atmospheric_model():
    # Test altitude in m
    altitude_m = 450000
    
    # Convert altitude to nearest multiple of 20 km
    #   This will round down to be conservative
    altitude_km = round(altitude_m / 1000 / 20) * 20

    # Get the min and max altitudes from the data
    min_altitude = min(low_solar_activity.keys())
    max_altitude = max(low_solar_activity.keys())

    # Clamp the altitude to the valid range
    test_altitude = max(min_altitude, min(max_altitude, altitude_km))
   

    # Print atmosphere data for all three solar activity levels at the test altitude
    print(f"Atmosphere data at {test_altitude} km altitude:")
    
    print("\nLow Solar Activity:")
    print(f"Temperature: {low_solar_activity[test_altitude]['temp']} K")
    print(f"Density: {low_solar_activity[test_altitude]['density']} kg/m³")
    print(f"Pressure: {low_solar_activity[test_altitude]['pressure']} Pa")
    print(f"Molecular Weight: {low_solar_activity[test_altitude]['mol_wt']}")

    print("\nModerate Solar Activity:")
    print(f"Temperature: {moderate_solar_activity[test_altitude]['temp']} K")
    print(f"Density: {moderate_solar_activity[test_altitude]['density']} kg/m³")
    print(f"Pressure: {moderate_solar_activity[test_altitude]['pressure']} Pa")
    print(f"Molecular Weight: {moderate_solar_activity[test_altitude]['mol_wt']}")

    print("\nHigh Solar Activity:")
    print(f"Temperature: {high_solar_activity[test_altitude]['temp']} K")
    print(f"Density: {high_solar_activity[test_altitude]['density']} kg/m³")
    print(f"Pressure: {high_solar_activity[test_altitude]['pressure']} Pa")
    print(f"Molecular Weight: {high_solar_activity[test_altitude]['mol_wt']}")
