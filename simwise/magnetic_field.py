import os
import igrf
import sys
from datetime import datetime

def magnetic_field(lat, lon, alt):
    """
    Calculate magnetic field components using IGRF-13 model.
    
    Parameters:
    -----------
    lat : float
        Latitude in degrees (-90 to 90)
    lon : float
        Longitude in degrees (-180 to 180)
    alt : float
        Altitude in kilometers above Earth's surface
        
    Returns:
    --------
    xarray.Dataset
        Dataset containing magnetic field components
    """
    # First ensure IGRF is built
    try:
        igrf.build()
    except Exception as e:
        print(f"Error building IGRF: {str(e)}")
        return None

    try:
        # Get current date in the format IGRF expects
        date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate magnetic field
        B = igrf.igrf(date, glat=lat, glon=lon, alt_km=alt)
        return B
        
    except Exception as e:
        print(f"Error calculating magnetic field: {str(e)}")
        print(f"IGRF package path: {os.path.dirname(igrf.__file__)}")
        print(f"Python version: {sys.version}")
        return None

# Test function
# try:
#     result = magnetic_field(90, 0, 100)
#     if result is not None:
#         print(f"Magnetic field at 90°N, 0°E, 100km altitude:")
#         print(f"North component = {result['north'].item():.1f} nT")
#         print(f"East component = {result['east'].item():.1f} nT")
#         print(f"Down component = {result['down'].item():.1f} nT")
#         print(f"Total field = {result['total'].item():.1f} nT")
#         print(f"Inclination = {result['incl'].item():.1f} degrees")
#         print(f"Declination = {result['decl'].item():.1f} degrees")
# except Exception as e:
#     print(f"Failed to calculate magnetic field: {str(e)}")