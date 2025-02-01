import pytest
import numpy as np
from datetime import datetime, timedelta
import erfa

from simwise.math import frame_transforms
from simwise.utils.time import dt_utc_to_jd

def ecef_to_eci_IAU(eci_vector, dt_utc):
    # Compute UTC Julian Date
    jd_utc = erfa.dtf2d("UTC", dt_utc.year, dt_utc.month, dt_utc.day,
                        dt_utc.hour, dt_utc.minute, dt_utc.second)
    
    # Convert UTC to TAI
    tai1, tai2 = erfa.utctai(jd_utc[0], jd_utc[1])
    
    # Convert TAI to TT
    tt1, tt2 = erfa.taitt(tai1, tai2)
    
    # For accurate results, replace dut1 with actual UT1-UTC value
    dut1 = 0.0
    ut11, ut12 = erfa.utcut1(jd_utc[0], jd_utc[1], dut1)
    
    # Polar motion coordinates in radians
    xp = 0.0
    yp = 0.0
    
    # Compute celestial-to-terrestrial transformation matrix
    rc2t = erfa.c2t06a(tt1, tt2, ut11, ut12, xp, yp)
    
    # Transform ECI vector to ECEF
    return np.dot(rc2t, eci_vector)

# Test fixtures
@pytest.fixture
def j2000_epoch():
    return datetime(2000, 1, 1, 12, 0, 0)

@pytest.fixture
def test_vectors():
    """Return a set of test vectors to use across multiple tests"""
    return {
        'zero': np.zeros(3),
        'unit_x': np.array([1.0, 0.0, 0.0]),
        'unit_y': np.array([0.0, 1.0, 0.0]),
        'unit_z': np.array([0.0, 0.0, 1.0]),
        'arbitrary': np.array([1.0, 2.0, 3.0])
    }

def test_eci_ecef_roundtrip(j2000_epoch, test_vectors):
    """Test that converting from ECI to ECEF and back returns original vector"""
    for name, vector in test_vectors.items():
        # Convert ECI to ECEF
        ecef = frame_transforms.eci_to_ecef(vector, j2000_epoch)
        # Convert back to ECI
        eci = frame_transforms.ecef_to_eci(ecef, j2000_epoch)
        # Check if we got our original vector back (with reasonable tolerance)
        np.testing.assert_allclose(vector, eci, rtol=1e-8, atol=1e-8,
            err_msg=f"Roundtrip conversion failed for {name} vector")

def test_rotation_matrix_properties():
    """Test mathematical properties of the rotation matrix"""
    test_time = datetime(2020, 1, 1)
    jd = dt_utc_to_jd(test_time)
    
    # Get rotation matrix
    R = frame_transforms.rotation_matrix(jd)
    
    # Test orthogonality: R * R^T should be identity
    np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=1e-8, atol=1e-8,
        err_msg="Rotation matrix is not orthogonal")
    
    # Test determinant should be ~1 for proper rotation
    np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=1e-8, atol=1e-8,
        err_msg="Rotation matrix determinant is not 1")

def test_velocity_transformation(j2000_epoch):
    """Test velocity transformation along with position"""
    pos = np.array([42164.0, 0.0, 0.0])  # Approximately GEO radius in km
    vel = np.array([0.0, 3.075, 0.0])    # Approximate GEO velocity in km/s
    
    # Convert both position and velocity
    ecef_pos, ecef_vel = frame_transforms.eci_to_ecef(pos, j2000_epoch, vel)
    
    # Convert back
    eci_pos, eci_vel = frame_transforms.ecef_to_eci(ecef_pos, j2000_epoch, ecef_vel)
    
    # Check roundtrip accuracy with appropriate tolerances
    np.testing.assert_allclose(pos, eci_pos, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(vel, eci_vel, rtol=1e-8, atol=1e-8)

def test_erfa_comparison():
    """Compare results with pyerfa library for validation"""
    test_time = datetime(2020, 1, 1)
    test_vector = np.array([6378.137, 0.0, 0.0])
    
    # Get ERFA result using full pipeline
    erfa_result = ecef_to_eci_IAU(test_vector, test_time)
    
    # Get our transformation
    our_result = frame_transforms.eci_to_ecef(test_vector, test_time)
    
    # Compare results with appropriate tolerance for different time handling
    np.testing.assert_allclose(erfa_result, our_result, rtol=1e-3)

def test_precession_nutation():
    """Test the precession/nutation computation"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    jd = dt_utc_to_jd(j2000)
    julian_century = (jd - 2451545.0) / 36525.0
    
    # Get celestial pole positions
    x, y = frame_transforms.compute_celestial_positions(julian_century)
    
    # Values should be small but not necessarily microscopic at J2000
    assert abs(x) < 1e-4
    assert abs(y) < 1e-4

def test_pn_table_generation():
    """Test precession/nutation table generation and interpolation"""
    epoch_jd = dt_utc_to_jd(datetime(2021, 1, 1))
    t_end_seconds = 20 * 24 * 3600  # 20 days
    
    # Generate table
    pn_table = frame_transforms.generate_ecef_pn_table(epoch_jd, t_end_seconds)
    
    # Check table properties
    assert len(pn_table) == 3  # Should cover 20 days with 10-day spacing
    
    # Check that all matrices are proper rotation matrices with appropriate tolerance
    for jd, matrix in pn_table.items():
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), rtol=1e-8, atol=1e-8)

def test_geosynchronous_satellite():
    """Test with known GEO satellite positions"""
    # GEO satellite should stay roughly over the same Earth longitude
    # For a GEO satellite at 0° longitude:
    # - At midnight UTC: satellite should be at [42164, 0, 0] km in ECEF
    # - 6 hours later: satellite should be at [0, 42164, 0] km in ECEF
    # - 12 hours later: satellite should be at [-42164, 0, 0] km in ECEF
    # - 18 hours later: satellite should be at [0, -42164, 0] km in ECEF
    
    geo_radius = 42164.0  # km, typical GEO radius
    
    # Test times for a full day
    base_time = datetime(2020, 1, 1, 0, 0, 0)  # midnight UTC
    test_times = [
        (base_time, [geo_radius, 0, 0]),               # 00:00 UTC
        (base_time + timedelta(hours=6), [0, geo_radius, 0]),  # 06:00 UTC
        (base_time + timedelta(hours=12), [-geo_radius, 0, 0]), # 12:00 UTC
        (base_time + timedelta(hours=18), [0, -geo_radius, 0])  # 18:00 UTC
    ]
    
    for time, expected_ecef in test_times:
        # Start with ECI position
        eci_pos = frame_transforms.ecef_to_eci(np.array(expected_ecef), time)
        
        # Convert back to ECEF
        ecef_pos = frame_transforms.eci_to_ecef(eci_pos, time)
        
        # Should match our expected ECEF position
        np.testing.assert_allclose(ecef_pos, expected_ecef, rtol=1e-3, atol=1,
            err_msg=f"GEO satellite position mismatch at {time}")
        
        # The magnitude should definitely stay constant
        np.testing.assert_allclose(np.linalg.norm(ecef_pos), geo_radius, rtol=1e-8,
            err_msg=f"GEO radius changed at {time}")

def test_long_term_stability():
    """Test stability of transformations over a long period"""
    test_vector = np.array([7000.0, 0.0, 0.0])
    start_date = datetime(2000, 1, 1)
    
    # Test at 30-day intervals over 5 years
    for days in range(0, 5*365, 30):
        test_time = start_date + timedelta(days=days)
        
        # Both transformations should remain stable
        result = frame_transforms.eci_to_ecef(test_vector, test_time)
        assert np.all(np.isfinite(result))
        assert np.allclose(np.linalg.norm(result), np.linalg.norm(test_vector), rtol=1e-8)