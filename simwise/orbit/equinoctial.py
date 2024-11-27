import numpy as np
from simwise.constants import *

def coe2mee(elements):
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

def mee2coe(elements):
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




def mee_dynamics(elements, mu, dt, f_perturbation):
    """ Do rigid body dyamics of a body in MEE

    Args:
        elements (_type_): _description_
        mu (_type_): _description_
        dt (_type_): _description_
        f (_type_): _description_
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
    s = (1+h**2+k**2)**0.5
    
    A = np.array([
        [0, ((2*p)/ω)*(p/mu)**0.5, 0],
        [((p/mu)**0.5)*np.sin(L), ((p/mu)**0.5)*(1/ω)*((ω+1)*np.cos(L) + f), -(p/mu)**0.5*(g/ω)*(h*np.sin(L) - k*np.cos(L))],
        [-((p/mu)**0.5)*np.cos(L), ((p/mu)**0.5)*(1/ω)*((ω+1)*np.sin(L)+g), ((p/mu)**0.5)*(f/ω)*(h*np.sin(L) - k*np.cos(L))],
        [0, 0, ((p/mu)**0.5)*(s**2*np.cos(L))/(2*ω)],
        [0, 0, ((p/mu)**0.5)*(s**2*np.sin(L))/(2*ω)],
        [0, 0, ((p/mu)**0.5)*(1/ω)*(h*np.sin(L) - k*np.cos(L))]
    ])

    b = np.array([
        0,
        0,
        0,
        0,
        0,
        np.sqrt(mu*p)*(ω/p)**2
    ])
    # print(A, f)

    return A @ f_perturbation + b


def mee_to_rv(mee, mu):
    # Does NOT work for elliptical orbits
    p, f, g, h, k, L = mee
    
    s2 = 1 + h**2 + k**2
    w = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / w
    
    # Position equations remain the same
    x = r * ((1 - h**2 + k**2) * np.cos(L) + 2*h*k*np.sin(L) - k)
    y = r * ((1 + h**2 - k**2) * np.sin(L) + 2*h*k*np.cos(L) + h)
    z = 2*r * (h*np.sin(L) - k*np.cos(L))
    
    sqrt_mu_p = np.sqrt(mu / p)
    
    # Modified velocity equations to handle equatorial case
    vx = -sqrt_mu_p * (-f + g + np.sin(L))  # Added np.sin(L) term
    vy = -sqrt_mu_p * (-g - f - np.cos(L))  # Added -np.cos(L) term
    vz = 2*sqrt_mu_p * (f*h*np.sin(L) - g*k*np.cos(L) - (f*k + g*h)/s2)
    
    r_vec = np.array([x, y, z])
    v_vec = np.array([vx, vy, vz])
    
    return r_vec, v_vec


def calculate_altitude(r_vec, r_earth):
    """Calculate altitude above Earth's surface."""
    return np.linalg.norm(r_vec) - r_earth


def calculate_trn_velocity(r_vec, v_vec):
    """Calculate velocity components in the Tangential, Radial, Normal (TRN) frame."""
    r_unit = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    n_unit = h_vec / np.linalg.norm(h_vec)
    t_unit = np.cross(n_unit, r_unit)
    
    v_radial = np.dot(v_vec, r_unit)
    v_tangential = np.dot(v_vec, t_unit)
    v_normal = np.dot(v_vec, n_unit)
    
    return np.array([v_tangential, v_radial, v_normal])


def get_velocity_vector_TRN(elements, mu=MU_EARTH, element_type='mee'):
    """
    Calculate velocity vector from orbital elements in TRN coordinates.
    
    Args:
        elements (np.ndarray): Orbital elements (MEE or COE)
        mu (float): Gravitational parameter (default is Earth's mu)
        element_type (str): 'mee' for Modified Equinoctial Elements, 'coe' for Classical Orbital Elements
    
    Returns:
        np.ndarray: Velocity vector in TRN coordinates [v_tangential, v_radial, v_normal]
    """
    if element_type == 'coe':
        mee = coe2mee(elements)
    elif element_type == 'mee':
        mee = elements
    else:
        raise ValueError("Invalid element_type. Use 'mee' or 'coe'.")
    
    r_vec, v_vec = mee_to_rv(mee, mu)
    
    return calculate_trn_velocity(r_vec, v_vec)

def get_altitude(elements, R_earth=RADIUS_OF_EARTH, mu=MU_EARTH, element_type='mee'):
    """
    Calculate altitude from orbital elements.
    
    Args:
        elements (np.ndarray): Orbital elements (MEE or COE)
        R_earth (float): Earth's radius (default is Earth's mean radius)
        mu (float): Gravitational parameter (default is Earth's mu)
        element_type (str): 'mee' for Modified Equinoctial Elements, 'coe' for Classical Orbital Elements
    
    Returns:
        float: Altitude above Earth's surface in [m]
    """
    if element_type == 'coe':
        mee = coe2mee(elements)
    elif element_type == 'mee':
        mee = elements
    else:
        raise ValueError("Invalid element_type. Use 'mee' or 'coe'.")
    
    r_vec, _ = mee_to_rv(mee, mu)
    
    return np.linalg.norm(r_vec) - R_earth

def get_position_vector(elements, mu=MU_EARTH, element_type='mee'):
    """
    Calculate position vector from orbital elements.
    
    Args:
        elements (np.ndarray): Orbital elements (MEE or COE)
        mu (float): Gravitational parameter (default is Earth's mu)
        element_type (str): 'mee' for Modified Equinoctial Elements, 'coe' for Classical Orbital Elements
    
    Returns:
        np.ndarray: Position vector in the inertial frame [x, y, z] in [m]
    """
    if element_type == 'coe':
        mee = coe2mee(elements)
    elif element_type == 'mee':
        mee = elements
    else:
        raise ValueError("Invalid element_type. Use 'mee' or 'coe'.")
    
    r_vec, _ = mee_to_rv(mee, mu)
    
    return r_vec


# Test Cases:
def init_circular_orbit(altitude=450e3):
    """
    Initialize Classical Orbital Elements (COE) for a circular polar orbit.

    Args:
        altitude (float): Orbit altitude in meters. Default is 450 km.

    Returns:
        np.ndarray: COE [a, e, i, Ω, ω, θ]
    """
    a = RADIUS_OF_EARTH + altitude  # Semi-major axis
    e = 0.0  # Eccentricity (circular orbit)
    i = 0.0  # Inclination (90 degrees for polar orbit)
    Ω = np.pi / 2  # Right ascension of the ascending node (arbitrary for polar orbit)
    ω = 0.0  # Argument of perigee (undefined for circular orbit, set to 0)
    θ = 0.0  # True anomaly (arbitrary for circular orbit)

    return np.array([a, e, i, Ω, ω, θ])

def init_elliptical_orbit():
    # Test case: Highly elliptical equatorial orbit
    # Perigee: 6678 km (300km altitude)
    # Apogee: 42164 km (GEO altitude)
    # Equatorial (i=0), starting at perigee
    
    rp = 6678000  # perigee radius in meters
    ra = 42164000  # apogee radius in meters
    
    # Classical orbital elements
    a = (rp + ra) / 2  # semi-major axis
    e = (ra - rp) / (ra + rp)  # eccentricity
    i = 0.0  # inclination
    Ω = 0.0  # RAAN
    ω = 0.0  # argument of perigee
    θ = 0.0  # true anomaly (starting at perigee)
    
    return np.array([a, e, i, Ω, ω, θ])

# Example usage
if __name__ == "__main__":
    # coe = init_circular_orbit()
    coe = init_elliptical_orbit()

    # vel_trn_coe = get_velocity_vector_TRN(coe, MU_EARTH, 'coe')
    # alt_coe = get_altitude(coe, RADIUS_OF_EARTH, MU_EARTH, 'coe')
    # pos_coe = get_position_vector(coe, MU_EARTH, 'coe')
    
    # print("Using Classical Orbital Elements:")
    # print(f"Velocity TRN components (m/s): Tangential = {vel_trn_coe[0]:.2f}, Radial = {vel_trn_coe[1]:.2f}, Normal = {vel_trn_coe[2]:.2f}")
    # print(f"Altitude: {alt_coe:.2f} m")
    # print(f"Position vector (m): [{pos_coe[0]:.2f}, {pos_coe[1]:.2f}, {pos_coe[2]:.2f}]")

    # # Calculate orbital velocity for verification
    # r = RADIUS_OF_EARTH + 450e3
    # v_orbital = np.sqrt(MU_EARTH / r)
    # print(f"\nCalculated orbital velocity: {v_orbital:.2f} m/s")

    # Convert to MEE and repeat calculations
    mee = coe2mee(coe)
    vel_trn_mee = get_velocity_vector_TRN(mee, MU_EARTH, 'mee')
    alt_mee = get_altitude(mee, RADIUS_OF_EARTH, MU_EARTH, 'mee')
    pos_mee = get_position_vector(mee, MU_EARTH, 'mee')
    
    print("\nUsing Modified Equinoctial Elements:")
    print(f"Velocity TRN components (m/s): Tangential = {vel_trn_mee[0]:.2f}, Radial = {vel_trn_mee[1]:.2f}, Normal = {vel_trn_mee[2]:.2f}")
    print(f"Altitude: {alt_mee:.2f} m")
    print(f"Position vector (m): [{pos_mee[0]:.2f}, {pos_mee[1]:.2f}, {pos_mee[2]:.2f}]")