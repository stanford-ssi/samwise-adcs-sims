import os
import igrf
import sys
import numpy as np
from datetime import datetime
from simwise.utils.time import jd_to_dt_utc
from simwise.math.coordinate_transforms import END_to_ECEF

# First ensure IGRF is built
# try:
#     igrf.build()
# except AttributeError as e:
#     igrf.base.build()
# except Exception as e:
#     print(f"Error building IGRF: {str(e)}")
#     raise e

def magnetic_field(lla_wgs84, jd):
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
    lat, lon, alt = lla_wgs84

    try:
        # Get current date in the format IGRF expects
        date = jd_to_dt_utc(jd).strftime('%Y-%m-%d')
        
        # Calculate magnetic field
        B = igrf.igrf(date, glat=lat, glon=lon, alt_km=alt/1e3)
        B_END = np.array((float(B.east), float(B.north), float(B.down)))
        B_ECEF = END_to_ECEF(B_END, lla_wgs84)
        
        return B_ECEF
        
    except Exception as e:
        print(f"Error calculating magnetic field: {str(e)}")
        print(f"IGRF package path: {os.path.dirname(igrf.__file__)}")
        print(f"Python version: {sys.version}")
        return None
    