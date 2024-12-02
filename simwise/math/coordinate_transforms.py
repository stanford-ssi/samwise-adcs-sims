import numpy as np
from simwise import constants

def ECEF_to_topocentric(position):
    """Convert state vector to latitude, longitude, and altitude.

    Args:
        rv (_type_): _description_
    """
    # WGS84 constants
    a = 6378137.0  # Semi-major axis (meters)
    f = 1 / 298.257223563  # Flattening
    b = a * (1 - f)  # Semi-minor axis
    e = np.sqrt(1 - b**2 / a**2)  # Eccentricity

    # Extract position components
    x, y, z = position

    # Longitude
    longitude = np.arctan2(y, x)

    # Calculate latitude
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    latitude = np.arctan2(z + e**2 * b * np.sin(theta)**3, p - a * e**2 * np.cos(theta)**3)

    # Calculate altitude
    N = a / np.sqrt(1 - e**2 * np.sin(latitude)**2)
    altitude = p / np.cos(latitude) - N

    return np.degrees(latitude), np.degrees(longitude), altitude

def coe_to_rv(coe, mu = constants.MU_EARTH):
    """
    Convert Classical Orbital Elements (COE) to State Vectors (RV)
    
    Parameters:
    coe : array-like
        Classical Orbital Elements in the form [a, e, i, Ω, ω, ν]
        where:
        a: semi-major axis (km)
        e: eccentricity
        i: inclination (degrees)
        Ω: right ascension of the ascending node (degrees)
        ω: argument of periapsis (degrees)
        ν: true anomaly (degrees)
    mu : float
        Gravitational parameter (km^3/s^2)
    
    Returns:
    array
        State vector [x, y, z, vx, vy, vz]
        where:
        x, y, z: position components (km)
        vx, vy, vz: velocity components (km/s)
    """
    a, e, i, Omega, omega, nu = coe
    
    # Convert degrees to radians
    i_rad = np.radians(i)
    Omega_rad = np.radians(Omega)
    omega_rad = np.radians(omega)
    nu_rad = np.radians(nu)
    
    # Calculate semi-latus rectum
    p = a * (1 - e**2)
    
    # Position in the perifocal coordinate system
    r_peri = np.array([
        p * np.cos(nu_rad) / (1 + e * np.cos(nu_rad)),
        p * np.sin(nu_rad) / (1 + e * np.cos(nu_rad)),
        0
    ])
    
    # Velocity in the perifocal coordinate system
    vx_peri = np.sqrt(mu / p) * (-np.sin(nu_rad))
    vy_peri = np.sqrt(mu / p) * (e + np.cos(nu_rad))
    vz_peri = 0
    v_peri = np.array([vx_peri, vy_peri, vz_peri])
    
    # Transformation matrix from perifocal to geocentric equatorial frame
    R = np.array([
        [np.cos(Omega_rad) * np.cos(omega_rad) - np.sin(Omega_rad) * np.sin(omega_rad) * np.cos(i_rad),
         -np.cos(Omega_rad) * np.sin(omega_rad) - np.sin(Omega_rad) * np.cos(omega_rad) * np.cos(i_rad),
         np.sin(Omega_rad) * np.sin(i_rad)],
        [np.sin(Omega_rad) * np.cos(omega_rad) + np.cos(Omega_rad) * np.sin(omega_rad) * np.cos(i_rad),
         -np.sin(Omega_rad) * np.sin(omega_rad) + np.cos(Omega_rad) * np.cos(omega_rad) * np.cos(i_rad),
         -np.cos(Omega_rad) * np.sin(i_rad)],
        [np.sin(omega_rad) * np.sin(i_rad),
         np.cos(omega_rad) * np.sin(i_rad),
         np.cos(i_rad)
        ]
    ])
    
    # Convert position and velocity to geocentric equatorial frame
    r = R @ r_peri
    v = R @ v_peri
    
    return np.concatenate((r, v))


def coe_to_mee(elements):
    """Transform classic (keplerian) orbital elements to modified equinoctial elements.

    Args:
        elements (np.ndarray): orbital elements of form [a, e, i, Ω, ω, θ]
    """
    a = elements[0]
    e = elements[1]
    i = elements[2]
    Ω = elements[3]
    ω = elements[4]
    θ = elements[5]

    p = a * (1 - e**2)
    f = e * np.cos(ω + θ)
    g = e * np.sin(ω + θ)
    h = np.tan(i / 2) * np.cos(Ω)
    k = np.tan(i / 2) * np.sin(Ω)
    L = Ω + ω + θ
    
    
    return np.array([p, f, g, h, k, L])

def mee_to_coe(elements):
    """Transform modified equinoctial elements to classic (keplerian) orbital elements.

    Args:
        elements (_type_): _description_
    """
    p = elements[0]
    f = elements[1]
    g = elements[2]
    h = elements[3]
    k = elements[4]
    L = elements[5]

    a = p / (1 - f**2 - g**2)
    e = np.sqrt(f**2 + g**2)
    i = np.arctan2(2 * np.sqrt(h**2 + k**2), 1 - h**2 - k**2)
    Ω = np.arctan2(k, h)
    ω = np.arctan2(g, f) - L
    θ = L - Ω - ω
    return np.array([a, e, i, Ω, ω, θ])