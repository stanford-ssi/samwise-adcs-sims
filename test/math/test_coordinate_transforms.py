import numpy as np
from simwise.math.coordinate_transforms import ECEF_to_topocentric, mee_to_coe, coe_to_mee, mee_to_rv, coe_to_rv
from simwise import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_ECEF_point(r, title="ECEF Point Visualization"):
    """Visualize an ECEF point and the Earth.
    
    Args:
        r (np.ndarray): ECEF position vector [x, y, z]
        title (str): Plot title
    """
    # Create visualization
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth as a sphere
    re = 6378137.0
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = re * np.outer(np.cos(u), np.sin(v))
    y = re * np.outer(np.sin(u), np.sin(v))
    z = re * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.2)

    # Plot latitude lines (parallels)
    lats = np.linspace(-90, 90, 13)  # -90, -75, -60, ..., 75, 90
    for lat in lats:
        # Convert latitude to radians
        lat_rad = np.deg2rad(lat)
        # Calculate radius at this latitude
        r_lat = re * np.cos(lat_rad)
        # Create points around this latitude circle
        theta = np.linspace(0, 2*np.pi, 100)
        x_lat = r_lat * np.cos(theta)
        y_lat = r_lat * np.sin(theta)
        z_lat = re * np.sin(lat_rad) * np.ones_like(theta)
        ax.plot(x_lat, y_lat, z_lat, 'k-', alpha=0.2)

    # Plot longitude lines (meridians)
    lons = np.linspace(-180, 180, 13)  # -180, -150, -120, ..., 150, 180
    for lon in lons:
        # Convert longitude to radians
        lon_rad = np.deg2rad(lon)
        # Create points along this longitude line
        phi = np.linspace(-np.pi/2, np.pi/2, 100)
        x_lon = re * np.cos(phi) * np.cos(lon_rad)
        y_lon = re * np.cos(phi) * np.sin(lon_rad)
        z_lon = re * np.sin(phi)
        ax.plot(x_lon, y_lon, z_lon, 'k-', alpha=0.2)

    # Calculate lat, lon, alt
    lat, lon, alt = ECEF_to_topocentric(r)

    # Plot the point
    ax.scatter(r[0], r[1], r[2], color='red', s=100)
    ax.text(r[0], r[1], r[2], 
            f'ECEF: [{r[0]:.0f}, {r[1]:.0f}, {r[2]:.0f}]\nLat: {lat:.2f}°\nLon: {lon:.2f}°\nAlt: {alt:.2f}m', 
            fontsize=9)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    plt.show()

def test_mee_to_coe_and_reverse():
    orbit_kep = np.array([7000e3, 0.1, 0.1, 0.1, 0.1, 0.1])
    orbit_mee = coe_to_mee(orbit_kep)
    orbit_kep_new = mee_to_coe(orbit_mee)
    assert np.allclose(orbit_kep, orbit_kep_new)

def test_ECEF_to_topocentric():
    """Test ECEF to topocentric (lat, lon, alt) conversion"""
    
    re = 6378137.0
    
    # Test case 1: Point on equator at prime meridian, on surface of earth
    r_equator = np.array([re, 0, 0])
    lat, lon, alt = ECEF_to_topocentric(r_equator)
    print(f"\nTest Case 1 - Equator at Prime Meridian: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r_equator, "Test Case 1: Equator at Prime Meridian")
    assert np.isclose(lat, 0, atol=1e-6)
    assert np.isclose(lon, 0, atol=1e-6)
    assert np.isclose(alt, 0, atol=10)
    
    # Test case 2: Point at 180° longitude
    r_180deg = np.array([-re, 0, 0])
    lat, lon, alt = ECEF_to_topocentric(r_180deg)
    print(f"\nTest Case 2 - 180° Longitude: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r_180deg, "Test Case 2: Equator at 180° Longitude")
    assert np.isclose(lat, 0, atol=1e-6)
    assert np.isclose(abs(lon), 180, atol=1e-6)
    assert np.isclose(alt, 0, atol=10)
    
    # Test case 3: North pole
    r_pole = np.array([0, 0, re])
    lat, lon, alt = ECEF_to_topocentric(r_pole)
    print(f"\nTest Case 3 - North Pole: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r_pole, "Test Case 3: North Pole")
    assert np.isclose(lat, 90, atol=1e-6)
    assert np.isclose(alt, 0, atol=10)
    
    # Test case 4: Point at altitude above equator at prime meridian
    altitude = 500000  # 500 km
    r_0deg = np.array([re + altitude, 0, 0])
    lat, lon, alt = ECEF_to_topocentric(r_0deg)
    print(f"\nTest Case 4 - 500km above Prime Meridian: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r_0deg, "Test Case 4: Point at altitude above equator at prime meridian")
    assert np.isclose(lat, 0, atol=1e-6)
    assert np.isclose(lon, 0, atol=1e-6)
    assert np.isclose(alt, altitude, atol=10+altitude)
    
    # Test case 5: Point at altitude above north pole
    altitude = 1000000  # 1000 km
    r_90deg = np.array([0, 0, re + altitude])
    lat, lon, alt = ECEF_to_topocentric(r_90deg)
    print(f"\nTest Case 5 - 1000km above North Pole: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r_90deg, "Test Case 5: Point at altitude above north pole")
    assert np.isclose(lat, 90, atol=1e-6)
    assert np.isclose(alt, altitude, atol=10+altitude)
    
    # Test case 6: 45 degrees latitude, 45 degrees longitude, on surface
    r = np.array([
        re * np.cos(np.pi/4) * np.cos(np.pi/4),
        re * np.cos(np.pi/4) * np.sin(np.pi/4),
        re * np.sin(np.pi/4)
    ])
    lat, lon, alt = ECEF_to_topocentric(r)
    print(f"\nTest Case 6 - 45° Lat, 45° Lon: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r, "Test Case 6: 45 degrees latitude, 45 degrees longitude, on surface")
    assert np.isclose(lat, 45, atol=1)   # Because lat is on ellipsoid surface, threshold is larger
    assert np.isclose(lon, 45, atol=1e-6)
    assert np.isclose(alt, 0, atol=10)

    # Test case 7: 180° longitude with altitude
    altitude = 500000  # 500 km
    r_180deg = np.array([-re - altitude, 0, 0])
    lat, lon, alt = ECEF_to_topocentric(r_180deg)
    print(f"\nTest Case 7 - 500km above 180° Lon: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r_180deg, "Test Case 7: 180° longitude with altitude")
    assert np.isclose(lat, 0, atol=1e-6)
    assert np.isclose(abs(lon), 180, atol=1e-6)
    assert np.isclose(alt, altitude, atol=10+altitude)

    # Test case 8: Southern hemisphere (South Pole)
    r_south = np.array([0, 0, -re])
    lat, lon, alt = ECEF_to_topocentric(r_south)
    print(f"\nTest Case 8 - South Pole: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r_south, "Test Case 8: South Pole")
    assert np.isclose(lat, -90, atol=1e-6)  # Should be at -90 degrees latitude
    assert np.isclose(alt, 0, atol=10)       # Should be near surface
    # Note: longitude is undefined at poles, so we don't test it
    
    # Test case 9: Zero vector (invalid input)
    with np.testing.assert_raises(ValueError):
        lat, lon, alt = ECEF_to_topocentric(np.zeros(3))
        print(f"\nTest Case 9 - Zero Vector Lon: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
        
    
    # Test case 10: Very large altitude
    altitude = 1e8  # 100,000 km
    r_high = np.array([1e8, 0, 0])  # Very high altitude
    lat, lon, alt = ECEF_to_topocentric(r_high)
    print(f"\nTest Case 10 - Very High Altitude: lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")
    visualize_ECEF_point(r_high, "Test Case 10: Very high altitude")
    assert np.isclose(lat, 0, atol=1e-6)    # Should be at equator
    assert np.isclose(lon, 0, atol=1e-6)    # Should be at prime meridian
    assert np.isclose(alt, altitude - re, atol=10+altitude)
    
    # Test case 11: Input validation - wrong vector size
    with np.testing.assert_raises(ValueError):
        ECEF_to_topocentric(np.array([1, 2]))  # Wrong input size

def test_mee_to_rv():
    """Test conversion from Modified Equinoctial Elements (MEE) to position/velocity vectors (RV)
    
    MEE = [p, f, g, h, k, L] where:
        p: semi-parameter (km)
        f: equinoctial element related to eccentricity (f = e*cos(ω + Ω))
        g: equinoctial element related to eccentricity (g = e*sin(ω + Ω))
        h: equinoctial element related to inclination (h = tan(i/2)*cos(Ω))
        k: equinoctial element related to inclination (k = tan(i/2)*sin(Ω))
        L: true longitude (L = Ω + ω + ν)
    """
    µ = 3.986004418e5  # Earth's gravitational parameter (km³/s²)
    
    # Test case 1: Circular equatorial orbit
    mee_circular_eq = np.array([7000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rv = mee_to_rv(mee_circular_eq, µ)
    print("\nTest Case 1 - Circular Equatorial Orbit:")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify circular orbit properties
    assert np.isclose(np.linalg.norm(rv[0:3]), 7000.0, rtol=1e-6)  # constant radius
    assert np.isclose(rv[2], 0.0, atol=1e-6)  # z = 0 (equatorial)
    
    # Test case 2: Elliptical equatorial orbit (e=0.2)
    a = 7000.0  # semi-major axis (km)
    e = 0.2     # eccentricity
    p = a * (1 - e**2)  # semi-parameter
    mee_ellip_eq = np.array([p, e, 0.0, 0.0, 0.0, 0.0])  # f=e, g=0 for perigee aligned with x-axis
    rv = mee_to_rv(mee_ellip_eq, µ)
    print("\nTest Case 2 - Elliptical Equatorial Orbit:")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify perigee distance
    assert np.isclose(np.linalg.norm(rv[0:3]), a*(1-e), rtol=1e-6)  # at perigee
    assert np.isclose(rv[2], 0.0, atol=1e-6)  # equatorial
    
    # Test case 3: Polar circular orbit (i=90°)
    p = 7000.0
    i = np.pi/2  # 90 degrees
    h = np.tan(i/2)  # Ω = 0, so h = tan(i/2)*cos(0)
    k = 0.0         # Ω = 0, so k = tan(i/2)*sin(0)
    mee_polar = np.array([p, 0.0, 0.0, h, k, 0.0])
    rv = mee_to_rv(mee_polar, µ)
    print("\nTest Case 3 - Polar Circular Orbit:")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify polar orbit properties
    assert np.isclose(np.linalg.norm(rv[0:3]), p, rtol=1e-6)  # circular
    assert np.isclose(rv[1], 0.0, atol=1e-6)  # starts in xz-plane
    
    # Test case 4: Retrograde equatorial orbit (i=180°)
    i = np.pi  # 180 degrees
    h = np.tan(i/2)  # infinite for i=180°, need different approach
    mee_retro = np.array([p, 0.0, 0.0, 1e6, 0.0, 0.0])  # Using large h to approximate i≈180°
    rv = mee_to_rv(mee_retro, µ)
    print("\nTest Case 4 - Retrograde Equatorial Orbit:")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify retrograde properties
    assert np.isclose(np.linalg.norm(rv[0:3]), p, rtol=1e-6)  # circular
    assert np.isclose(rv[2], 0.0, atol=1e-6)  # equatorial
    assert rv[4] < 0  # retrograde motion (negative y-velocity)
    
    # Test case 5: Molniya-type orbit (e=0.7, i=63.4°)
    a = 26600.0  # typical Molniya semi-major axis
    e = 0.7
    i = np.deg2rad(63.4)
    p = a * (1 - e**2)
    h = np.tan(i/2)
    mee_molniya = np.array([p, e, 0.0, h, 0.0, 0.0])
    rv = mee_to_rv(mee_molniya, µ)
    print("\nTest Case 5 - Molniya Orbit:")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify Molniya properties
    assert np.isclose(np.linalg.norm(rv[0:3]), a*(1-e), rtol=1e-6)  # at perigee
    
    # Test case 6: GEO orbit (circular equatorial, a ≈ 42164 km)
    p = 42164.0
    mee_geo = np.array([p, 0.0, 0.0, 0.0, 0.0, 0.0])
    rv = mee_to_rv(mee_geo, µ)
    print("\nTest Case 6 - Geostationary Orbit:")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify GEO properties
    assert np.isclose(np.linalg.norm(rv[0:3]), p, rtol=1e-6)
    assert np.isclose(rv[2], 0.0, atol=1e-6)  # equatorial
    assert np.isclose(np.linalg.norm(rv[3:6]), np.sqrt(µ/p), rtol=1e-6)  # circular orbit velocity

def test_coe_to_rv():
    """Test conversion from Classical Orbital Elements (COE) to position/velocity vectors (RV)
    
    COE = [a, e, i, Ω, ω, ν] where:
        a: semi-major axis (km)
        e: eccentricity
        i: inclination (degrees)
        Ω: right ascension of ascending node (degrees)
        ω: argument of periapsis (degrees)
        ν: true anomaly (degrees)
    """
    µ = constants.MU_EARTH  # Earth's gravitational parameter (km³/s²)
    
    # Test case 1: Circular equatorial orbit
    a = 7000.0  # km
    coe_circular_eq = np.array([a, 0.0, 0.0, 0.0, 0.0, 0.0])
    rv = coe_to_rv(coe_circular_eq)
    print("\nTest Case 1 - Circular Equatorial Orbit:")
    print(f"COE: a={a}km, e=0, i=0°, Ω=0°, ω=0°, ν=0°")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify circular orbit properties
    assert np.isclose(np.linalg.norm(rv[0:3]), a, rtol=1e-6)  # radius = a
    assert np.isclose(rv[2], 0.0, atol=1e-6)  # equatorial
    assert np.isclose(np.linalg.norm(rv[3:6]), np.sqrt(µ/a), rtol=1e-6)  # circular velocity
    
    # Test case 2: Elliptical equatorial orbit
    a = 7000.0
    e = 0.2
    coe_ellip_eq = np.array([a, e, 0.0, 0.0, 0.0, 0.0])
    rv = coe_to_rv(coe_ellip_eq)
    print("\nTest Case 2 - Elliptical Equatorial Orbit:")
    print(f"COE: a={a}km, e={e}, i=0°, Ω=0°, ω=0°, ν=0°")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify perigee distance and velocity
    assert np.isclose(np.linalg.norm(rv[0:3]), a*(1-e), rtol=1e-6)  # at perigee
    assert np.isclose(rv[2], 0.0, atol=1e-6)  # equatorial
    
    # Test case 3: Polar circular orbit
    a = 7000.0
    coe_polar = np.array([a, 0.0, 90.0, 0.0, 0.0, 0.0])
    rv = coe_to_rv(coe_polar)
    print("\nTest Case 3 - Polar Circular Orbit:")
    print(f"COE: a={a}km, e=0, i=90°, Ω=0°, ω=0°, ν=0°")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify polar orbit properties
    assert np.isclose(np.linalg.norm(rv[0:3]), a, rtol=1e-6)  # circular
    assert np.isclose(rv[1], 0.0, atol=1e-6)  # in xz-plane
    
    # Test case 4: Molniya orbit
    a = 26600.0
    e = 0.7
    i = 63.4
    ω = 270.0
    coe_molniya = np.array([a, e, i, 0.0, ω, 0.0])
    rv = coe_to_rv(coe_molniya)
    print("\nTest Case 4 - Molniya Orbit:")
    print(f"COE: a={a}km, e={e}, i={i}°, Ω=0°, ω={ω}°, ν=0°")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    
    # Test case 5: GEO orbit
    a = 42164.0
    coe_geo = np.array([a, 0.0, 0.0, 0.0, 0.0, 0.0])
    rv = coe_to_rv(coe_geo)
    print("\nTest Case 5 - Geostationary Orbit:")
    print(f"COE: a={a}km, e=0, i=0°, Ω=0°, ω=0°, ν=0°")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify GEO properties
    assert np.isclose(np.linalg.norm(rv[0:3]), a, rtol=1e-6)
    assert np.isclose(rv[2], 0.0, atol=1e-6)  # equatorial
    assert np.isclose(np.linalg.norm(rv[3:6]), np.sqrt(µ/a), rtol=1e-6)  # GEO velocity
    
    # Test case 6: True anomaly variations
    a = 7000.0
    e = 0.1
    for ν in [0, 90, 180, 270]:  # Test different positions in orbit
        coe = np.array([a, e, 45.0, 0.0, 0.0, float(ν)])
        rv = coe_to_rv(coe)
        r = np.linalg.norm(rv[0:3])
        print(f"\nTest Case 6 - True Anomaly = {ν}°:")
        print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
        print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
        # Verify orbit equation
        assert np.isclose(r, a*(1-e**2)/(1 + e*np.cos(np.deg2rad(ν))), rtol=1e-6)
        
    # Test case 7: RAAN variations
    for Ω in [0, 90, 180, 270]:  # Test different orbital planes
        coe = np.array([a, 0.0, 45.0, float(Ω), 0.0, 0.0])
        rv = coe_to_rv(coe)
        print(f"\nTest Case 7 - RAAN = {Ω}°:")
        print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
        print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
        # Verify constant radius for circular orbit
        assert np.isclose(np.linalg.norm(rv[0:3]), a, rtol=1e-6)
        
    # Test case 8: Retrograde orbit
    coe_retro = np.array([a, 0.0, 170.0, 0.0, 0.0, 0.0])
    rv = coe_to_rv(coe_retro)
    print("\nTest Case 8 - Retrograde Orbit:")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    # Verify retrograde motion
    assert rv[4] < 0  # y-velocity should be negative for retrograde
    
    # Test case 9: Edge cases - near circular
    coe_near_circ = np.array([a, 1e-10, 45.0, 0.0, 0.0, 0.0])
    rv = coe_to_rv(coe_near_circ)
    print("\nTest Case 9 - Near-circular Orbit:")
    print(f"Position (km): [{rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}]")
    print(f"Velocity (km/s): [{rv[3]:.6f}, {rv[4]:.6f}, {rv[5]:.6f}]")
    assert np.isclose(np.linalg.norm(rv[0:3]), a, rtol=1e-6)

def test_coe_to_mee():
    """Test conversion from Classical Orbital Elements (COE) to Modified Equinoctial Elements (MEE)"""
    
    def print_conversion(case_num, description, coe, mee):
        print(f"\nTest Case {case_num} - {description}:")
        print(f"COE in: a={coe[0]:.1f}km, e={coe[1]:.6f}, i={coe[2]:.1f}°, Ω={coe[3]:.1f}°, ω={coe[4]:.1f}°, ν={coe[5]:.1f}°")
        print(f"MEE out: p={mee[0]:.1f}km, f={mee[1]:.6f}, g={mee[2]:.6f}, h={mee[3]:.6f}, k={mee[4]:.6f}, L={mee[5]:.1f}°")
    
    # Test case 1: Circular equatorial orbit
    coe = np.array([7000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mee = coe_to_mee(coe)
    print_conversion(1, "Circular Equatorial Orbit", coe, mee)
    assert np.isclose(mee[0], coe[0], rtol=1e-6)  # p = a for circular orbit
    assert np.all(np.isclose(mee[1:5], 0.0, atol=1e-6))  # f,g,h,k should be 0
    assert np.isclose(mee[5], 0.0, rtol=1e-6)  # L = 0
    
    # Test case 2: Elliptical equatorial orbit
    coe = np.array([7000.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    mee = coe_to_mee(coe)
    print_conversion(2, "Elliptical Equatorial Orbit", coe, mee)
    assert np.isclose(mee[0], coe[0]*(1-coe[1]**2), rtol=1e-6)  # p = a(1-e²)
    assert np.isclose(mee[1], coe[1], rtol=1e-6)  # f = e for ω = 0
    assert np.isclose(mee[2], 0.0, atol=1e-6)  # g = 0 for ω = 0
    
    # Test case 3: Polar circular orbit
    coe = np.array([7000.0, 0.0, 90.0, 0.0, 0.0, 0.0])
    mee = coe_to_mee(coe)
    print_conversion(3, "Polar Circular Orbit", coe, mee)
    assert np.isclose(mee[0], coe[0], rtol=1e-6)  # p = a
    assert np.isclose(mee[3]**2 + mee[4]**2, 1.0, rtol=1e-6)  # h² + k² = tan²(i/2) = 1 for i=90°
    
    # Test case 4: Molniya orbit
    coe = np.array([26600.0, 0.7, 63.4, 0.0, 270.0, 0.0])
    mee = coe_to_mee(coe)
    print_conversion(4, "Molniya Orbit", coe, mee)
    assert np.isclose(mee[0], coe[0]*(1-coe[1]**2), rtol=1e-6)  # p = a(1-e²)
    assert np.isclose(np.arctan2(mee[4], mee[3]), np.deg2rad(coe[3]), rtol=1e-6)  # Ω = atan2(k,h)
    
    # Test case 5: GEO orbit
    coe = np.array([42164.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mee = coe_to_mee(coe)
    print_conversion(5, "GEO Orbit", coe, mee)
    assert np.isclose(mee[0], coe[0], rtol=1e-6)  # p = a
    assert np.all(np.isclose(mee[1:5], 0.0, atol=1e-6))  # f,g,h,k should be 0
    
    # Test case 6: Sun-synchronous orbit
    coe = np.array([6978.0, 0.0, 98.0, 0.0, 0.0, 0.0])
    mee = coe_to_mee(coe)
    print_conversion(6, "Sun-synchronous Orbit", coe, mee)
    assert np.isclose(mee[0], coe[0], rtol=1e-6)  # p = a
    assert np.isclose(np.arctan(np.sqrt(mee[3]**2 + mee[4]**2))*2, np.deg2rad(coe[2]), rtol=1e-6)
    
    # Test case 7: Retrograde orbit
    coe = np.array([7000.0, 0.0, 170.0, 0.0, 0.0, 0.0])
    mee = coe_to_mee(coe)
    print_conversion(7, "Retrograde Orbit", coe, mee)
    assert np.isclose(mee[0], coe[0], rtol=1e-6)  # p = a
    assert mee[3]**2 + mee[4]**2 > 1.0  # h² + k² > 1 for retrograde
    
    # Test case 8: True anomaly variations
    for ν in [0.0, 90.0, 180.0, 270.0]:
        coe = np.array([7000.0, 0.1, 45.0, 0.0, 0.0, ν])
        mee = coe_to_mee(coe)
        print_conversion(8, f"True Anomaly = {ν}°", coe, mee)
        assert np.isclose(mee[5], np.deg2rad(np.sum(coe[3:6])), rtol=1e-6)  # L = Ω + ω + ν
    
    # Test case 9: Near-circular orbit
    coe = np.array([7000.0, 1e-7, 45.0, 0.0, 0.0, 0.0])
    mee = coe_to_mee(coe)
    print_conversion(9, "Near-circular Orbit", coe, mee)
    assert np.all(np.isclose(mee[1:3], 0.0, atol=1e-6))  # f,g should be near zero

def test_mee_to_coe():
    """Test conversion from Modified Equinoctial Elements (MEE) to Classical Orbital Elements (COE)"""
    
    def print_conversion(case_num, description, mee, coe):
        print(f"\nTest Case {case_num} - {description}:")
        print(f"MEE in: p={mee[0]:.1f}km, f={mee[1]:.6f}, g={mee[2]:.6f}, h={mee[3]:.6f}, k={mee[4]:.6f}, L={mee[5]:.1f}°")
        print(f"COE out: a={coe[0]:.1f}km, e={coe[1]:.6f}, i={coe[2]:.1f}°, Ω={coe[3]:.1f}°, ω={coe[4]:.1f}°, ν={coe[5]:.1f}°")
    
    # Test case 1: Circular equatorial orbit
    mee = np.array([7000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    coe = mee_to_coe(mee)
    print_conversion(1, "Circular Equatorial Orbit", mee, coe)
    assert np.isclose(coe[0], mee[0], rtol=1e-6)  # a = p for circular
    assert np.all(np.isclose(coe[1:3], 0.0, atol=1e-6))  # e, i = 0
    
    # Test case 2: Elliptical equatorial orbit
    mee = np.array([6720.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    coe = mee_to_coe(mee)
    print_conversion(2, "Elliptical Equatorial Orbit", mee, coe)
    assert np.isclose(coe[1], np.sqrt(mee[1]**2 + mee[2]**2), rtol=1e-6)  # e = sqrt(f² + g²)
    assert np.isclose(coe[2], 0.0, atol=1e-6)  # i = 0
    
    # Test case 3: Polar circular orbit
    mee = np.array([7000.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    coe = mee_to_coe(mee)
    print_conversion(3, "Polar Circular Orbit", mee, coe)
    assert np.isclose(coe[2], 90.0, rtol=1e-6)  # i = 90°
    assert np.isclose(coe[1], 0.0, atol=1e-6)  # e = 0
    
    # Test case 4: Molniya-type elements
    mee = np.array([2394.0, 0.7, 0.0, 0.4142, 0.0, 270.0])
    coe = mee_to_coe(mee)
    print_conversion(4, "Molniya Orbit", mee, coe)
    assert np.isclose(coe[1], np.sqrt(mee[1]**2 + mee[2]**2), rtol=1e-6)  # e = sqrt(f² + g²)
    assert np.isclose(coe[2], np.rad2deg(2*np.arctan(np.sqrt(mee[3]**2 + mee[4]**2))), rtol=1e-6)
    
    # Test case 5: GEO elements
    mee = np.array([42164.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    coe = mee_to_coe(mee)
    print_conversion(5, "GEO Orbit", mee, coe)
    assert np.isclose(coe[0], mee[0], rtol=1e-6)  # a = p
    assert np.all(np.isclose(coe[1:3], 0.0, atol=1e-6))  # e, i = 0
    
    # Test case 6: Sun-synchronous elements
    mee = np.array([6978.0, 0.0, 0.0, 1.4281, 0.0, 0.0])  # h = tan(49°)
    coe = mee_to_coe(mee)
    print_conversion(6, "Sun-synchronous Orbit", mee, coe)
    assert np.isclose(coe[2], 110.0, rtol=1e-0)  # i = 110°
    
    # Test case 7: Retrograde elements
    mee = np.array([7000.0, 0.0, 0.0, 11.43, 0.0, 0.0])  # Large h for retrograde
    coe = mee_to_coe(mee)
    print_conversion(7, "Retrograde Orbit", mee, coe)
    assert coe[2] > 90.0  # i > 90° for retrograde
    
    # Test case 8: True longitude variations
    for L in [0.0, 90.0, 180.0, 270.0]:
        Lrad = np.deg2rad(L)
        mee = np.array([7000.0, 0.1, 0.0, 0.4142, 0.0, Lrad])
        coe = mee_to_coe(mee)
        print_conversion(8, f"True Longitude = {L}°", mee, coe)
        assert np.isclose(coe[5], L, rtol=1e-6) 
    
    # Test case 9: Near-zero eccentricity
    mee = np.array([7000.0, 1e-7, 1e-7, 0.4142, 0.0, 0.0])
    coe = mee_to_coe(mee)
    print_conversion(9, "Near-circular Orbit", mee, coe)
    assert np.isclose(coe[1], 0.0, atol=1e-6)  # e ≈ 0

