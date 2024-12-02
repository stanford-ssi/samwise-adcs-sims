import erfa
import numpy as np

from simwise.math.coordinate_transforms import ECEF_to_topocentric
from simwise import constants

def ecef_to_eci_IAU(eci_vector, dt_utc):
    # Compute UTC Julian Date
    jd_utc = erfa.dtf2d("UTC", dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second)
    
    # Convert UTC to TAI
    tai1, tai2 = erfa.utctai(jd_utc[0], jd_utc[1])
    
    # Convert TAI to TT
    tt1, tt2 = erfa.taitt(tai1, tai2)
    
    # For accurate results, replace dut1 with the actual UT1-UTC value in seconds
    dut1 = 0.0  # UT1 - UTC in seconds
    
    # Convert UTC to UT1
    ut11, ut12 = erfa.utcut1(jd_utc[0], jd_utc[1], dut1)
    
    # Polar motion coordinates in radians (replace with actual values for higher accuracy)
    xp = 0.0
    yp = 0.0
    
    # Compute celestial-to-terrestrial transformation matrix
    rc2t = erfa.c2t00b(tt1, tt2, ut11, ut12, xp, yp)
    
    # Transform ECI vector to ECEF
    return np.dot(rc2t, eci_vector)

def test_ECEF_to_topocentric():
    # Test ECEF to topocentric conversion
    ecef = np.array([7000e3, 0, 0])
    latitude, longitude, altitude = ECEF_to_topocentric(ecef)
    assert latitude == 0.0
    assert longitude == 0.0
    assert np.isclose(altitude, 7000e3 - constants.EARTH_RADIUS_M, atol=1e3)

    ecef = np.array([0, 7000e3, 0])
    latitude, longitude, altitude = ECEF_to_topocentric(ecef)
    assert latitude == 0.0
    assert longitude == 90.0
    assert np.isclose(altitude, 7000e3 - constants.EARTH_RADIUS_M, atol=1e3)

    ecef = np.array([-7000e3, 0, 0])
    latitude, longitude, altitude = ECEF_to_topocentric(ecef)
    assert latitude == 0.0
    assert longitude == 180.0
    assert np.isclose(altitude, 7000e3 - constants.EARTH_RADIUS_M, atol=1e3)

    ecef = np.array([0, -7000e3, 0])
    latitude, longitude, altitude = ECEF_to_topocentric(ecef)
    assert latitude == 0.0
    assert longitude == -90
    assert np.isclose(altitude, 7000e3 - constants.EARTH_RADIUS_M, atol=1e3)

    ecef = np.array([100, 0, 7000e3])
    latitude, longitude, altitude = ECEF_to_topocentric(ecef)
    assert np.isclose(latitude, 90, atol=1e-3)
    assert longitude == 0.0
    assert np.isclose(altitude, 7000e3 - constants.EARTH_RADIUS_M, atol=2.5e4)

    ecef = np.array([100, 0, -7000e3])
    latitude, longitude, altitude = ECEF_to_topocentric(ecef)
    assert np.isclose(latitude, -90, atol=1e-3)
    assert longitude == 0.0
    assert np.isclose(altitude, 7000e3 - constants.EARTH_RADIUS_M, atol=2.5e4)
    