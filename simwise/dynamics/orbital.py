from numba import jit
import numpy as np
from simwise.constants import *

@jit(nopython=True)
def mee_dynamics(elements, mu, dt, f_perturbation):
    """Compute rigid body dynamics in Modified Equinoctial Elements (MEE).

    Args:
        elements (np.ndarray): Orbital elements [p, f, g, h, k, L].
        mu (float): Gravitational parameter.
        dt (float): Time step (unused here, but kept for consistency).
        f_perturbation (np.ndarray): Perturbation forces [fr, ft, fn].

    Returns:
        np.ndarray: Time derivative of MEE.
    """
    # Extract elements
    p, f, g, h, k, L = elements

    # Precompute reusable terms
    cosL = np.cos(L)
    sinL = np.sin(L)
    w = 1 + f * cosL + g * sinL
    root_p_mu = np.sqrt(p / mu)
    inv_w = 1 / w
    s_2 = 1 + h**2 + k**2
    w_plus_1 = w + 1

    # Preallocate matrix A for faster computation (with numba)
    A = np.zeros((6, 3))

    # Fill matrix A directly
    A[0, 1] = 2 * p * inv_w
    A[1, 0] = sinL
    A[1, 1] = inv_w * (w_plus_1 * cosL + f)
    A[1, 2] = -(g * inv_w) * (h * sinL - k * cosL)
    A[2, 0] = -cosL
    A[2, 1] = inv_w * (w_plus_1 * sinL + g)
    A[2, 2] = (f * inv_w) * (h * sinL + k * cosL)
    A[3, 2] = (s_2 * cosL) / (2 * w)
    A[4, 2] = (s_2 * sinL) / (2 * w)
    A[5, 2] = (h * sinL - k * cosL) * inv_w

    # Scale A by root_p_mu
    A *= root_p_mu

    # Compute vector b
    b = np.zeros(6)
    b[5] = np.sqrt(mu / p**3) * w**2

    # Return dynamics
    return A @ f_perturbation + b



def get_altitude(rv, radius_earth=6371):
    """
    Calculate altitude from state vector
    
    Parameters:
    rv : array-like
        State vector [x, y, z, vx, vy, vz]
    radius_earth : float, optional
        Radius of the Earth in km (default is 6371 km)
    
    Returns:
    float
        Altitude in km
    """
    r = rv[:3]
    return np.linalg.norm(r) - radius_earth

def get_velocity(rv):
    """
    Calculate velocity magnitude from state vector
    
    Parameters:
    rv : array-like
        State vector [x, y, z, vx, vy, vz]
    
    Returns:
    float
        Velocity magnitude in km/s
    """
    v = rv[3:]
    return np.linalg.norm(v)