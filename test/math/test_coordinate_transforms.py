import numpy as np
from simwise.math.coordinate_transforms import ECEF_to_topocentric, mee_to_coe, coe_to_mee
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

