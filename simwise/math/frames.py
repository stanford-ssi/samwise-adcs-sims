import numpy as np

cos = np.cos
sin = np.sin

def precession_matrix(JD):
    """
    Compute the precession matrix for a given Julian Date (JD).
    """
    T = (JD - 2451545.0) / 36525.0  # Centuries since J2000.0

    # Precession angles in arcseconds (converted to radians)
    zeta = np.deg2rad((2306.2181 + 0.30188 * T + 0.017998 * T**2) / 3600.0)
    z = np.deg2rad((2306.2181 + 1.09468 * T + 0.018203 * T**2) / 3600.0)
    theta = np.deg2rad((2004.3109 - 0.42665 * T - 0.041833 * T**2) / 3600.0)

    # Precession matrix
    P = np.array([
        [cos(zeta) * cos(theta) * cos(z) - sin(zeta) * sin(z),
         -cos(zeta) * cos(theta) * sin(z) - sin(zeta) * cos(z),
         -cos(zeta) * sin(theta)],
        [sin(zeta) * cos(theta) * cos(z) + cos(zeta) * sin(z),
         -sin(zeta) * cos(theta) * sin(z) + cos(zeta) * cos(z),
         -sin(zeta) * sin(theta)],
        [sin(theta) * cos(z),
         -sin(theta) * sin(z),
         cos(theta)]
    ])
    return P

def nutation_matrix(JD):
    """
    Compute the nutation matrix for a given Julian Date (JD).
    """
    T = (JD - 2451545.0) / 36525.0  # Centuries since J2000.0

    # Nutation in longitude and obliquity (np.deg2rad)
    delta_psi = np.deg2rad(-17.20 * sin(np.deg2rad(125.04 - 1934.136 * T)) / 3600.0)
    delta_epsilon = np.deg2rad(9.20 * cos(np.deg2rad(125.04 - 1934.136 * T)) / 3600.0)

    # Mean obliquity of the ecliptic (arcseconds to np.deg2rad)
    epsilon_0 = np.deg2rad((84381.406 - 46.836769 * T - 0.0001831 * T**2) / 3600.0)

    # Nutation matrix
    N = np.array([
        [1, -delta_psi * cos(epsilon_0), -delta_psi * sin(epsilon_0)],
        [delta_psi * cos(epsilon_0), 1, -delta_epsilon],
        [delta_psi * sin(epsilon_0), delta_epsilon, 1]
    ])
    return N

def earth_rotation_matrix(JD):
    """
    Compute the Earth's rotation matrix for a given Julian Date (JD).
    """
    T = (JD - 2451545.0) / 36525.0  # Centuries since J2000.0

    # Greenwich Sidereal Time (degrees)
    GMST = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + \
           0.000387933 * T**2 - T**3 / 38710000.0
    GMST2 = 280.4606 + 360.9856473*(JD - 2451545.0)
    GMST = np.deg2rad(GMST % 360.0)  # Convert to np.deg2rad
    # print(GMST)

    # Earth rotation matrix
    R = np.array([
        [cos(GMST), sin(GMST), 0],
        [-sin(GMST), cos(GMST), 0],
        [0, 0, 1]
    ])
    return R

def ecliptic_to_equatorial():
    """
    Rotate the position vector from the ecliptic plane to the equatorial plane.
    """
    epsilon = np.radians(23.43928)  # Mean obliquity of the ecliptic at J2000
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon), -np.sin(epsilon)],
        [0, np.sin(epsilon), np.cos(epsilon)]
    ])
    return rotation_matrix