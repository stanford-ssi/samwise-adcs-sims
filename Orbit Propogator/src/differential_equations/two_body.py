import numpy as np
from ..constants import earth_mu, earth_radius, earth_j2


# state = [rx, ry, rz, vx, vy, vz]
# Acceleration due to Gravity
def TwoBodyODE(state: np.ndarray[float], mu: float) -> np.ndarray[float]:
    # Get the position vector from state
    r_vector = state[:3]

    # Calculate the acceleration vector
    a_vector = (-mu / np.linalg.norm(r_vector) ** 3) * r_vector

    # Return the derivative of the state
    return np.array([a_vector[0], a_vector[1], a_vector[2]])


def J2PertODE(state: np.ndarray[float]) -> np.ndarray[float]:
    r_norm = np.linalg.norm(state[:3])
    r_squared = r_norm**2
    z_squared = state[2] ** 2

    p = (3 * earth_j2 * earth_mu * (earth_radius**2)) / (2 * (r_squared**2))

    ax = ((5 * z_squared / r_squared) - 1) * (state[0] / r_norm)
    ay = ((5 * z_squared / r_squared) - 1) * (state[1] / r_norm)
    az = ((5 * z_squared / r_squared) - 3) * (state[2] / r_norm)

    return np.array([ax, ay, az]) * p


# state = [rx, ry, rz, vx, vy, vz]
# Differential equation for the state vector
def ODE(state: np.ndarray[float], mu: float) -> np.ndarray[float]:
    rx, ry, rz, vx, vy, vz = state

    state_dot = np.zeros(6)

    # Newton's Universal Law of Gravitation
    a = TwoBodyODE(state, mu)

    # J2 Perturbation
    a += J2PertODE(state)

    # Return the derivative of the state
    state_dot[:3] = [vx, vy, vz]
    state_dot[3:6] = a

    return state_dot
