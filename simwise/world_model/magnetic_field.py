import ppigrf
from datetime import datetime
from simwise.utils.time import jd_to_dt_utc

def magnetic_field(lla_wgs84, jd):
    """
    Calculate magnetic field components using IGRF-13 model via ppigrf.
    
    Parameters:
    -----------
    lla_wgs84 : tuple
        Tuple containing (latitude, longitude, altitude)
        latitude: float (-90 to 90 degrees)
        longitude: float (-180 to 180 degrees)
        altitude: float (meters above Earth's surface)
    jd : float
        Julian date
        
    Returns:
    --------
    tuple
        (east, north, down) components of the magnetic field in nanoTesla
    """
    lat, lon, alt = lla_wgs84
    
    try:
        # Convert Julian date to decimal year
        dt = jd_to_dt_utc(jd)
        date = datetime(2022, 3, 28)
        
        # Calculate magnetic field components
        # ppigrf.igrf() returns Be (east), Bn (north), Bu (up) in nT
        # Note: ppigrf expects altitude in kilometers
        Be, Bn, Bu = ppigrf.igrf(lon, lat, alt/1000, date) # returns east, north, up

        # Convert up component to down component
        Bd = -Bu
        
        return float(Be), float(Bn), float(Bd)
        
    except Exception as e:
        print(f"Error calculating magnetic field: {str(e)}")
        return None