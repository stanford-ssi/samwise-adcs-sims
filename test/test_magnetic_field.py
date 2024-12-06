from simwise.world_model.magnetic_field import magnetic_field

# Test function
# USE ECF
try:
    result = magnetic_field(90, 0, 100)
    if result is not None:
        print(f"Magnetic field at 90°N, 0°E, 100km altitude:")
        print(f"North component = {result['north'].item():.1f} nT")
        print(f"East component = {result['east'].item():.1f} nT")
        print(f"Down component = {result['down'].item():.1f} nT")
        print(f"Total field = {result['total'].item():.1f} nT")
except Exception as e:
    print(f"Failed to calculate magnetic field: {str(e)}")