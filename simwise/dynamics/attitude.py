import numpy as np
from simwise.math.quaternion import normalize_quaternion
from numba import jit


@jit(nopython=True)
def attitude_dynamics(x, dt, inertia_inv, inertia_diff, tau):
    """Apply quaternion dynamics and Euler's equations to state vector x.

    Args:
        x (np.ndarray): State vector of the form [q1, q2, q3, q4, Ω1, Ω2, Ω3].
        dt (float): Time interval.
        inertia_inv (np.ndarray): Inverse of the diagonal inertia tensor.
        inertia_diff (np.ndarray): Differences of inertia tensor components: [I2-I3, I3-I1, I1-I2].
        tau (np.ndarray): Torque vector in the body frame.
    """
    Q = normalize_quaternion(x[:4])
    Ω = x[4:]

    # Quaternion kinematics
    stm = np.array([
        [0,   -Ω[0], -Ω[1], -Ω[2]],
        [Ω[0],    0,  Ω[2], -Ω[1]],
        [Ω[1],-Ω[2],     0,  Ω[0]],
        [Ω[2], Ω[1], -Ω[0],     0]
    ])
    Q_dot = 0.5 * stm @ Q

    # Angular velocity dynamics
    Ω_dot = np.array([
        inertia_diff[0] * Ω[1] * Ω[2] + tau[0] * inertia_inv[0],
        inertia_diff[1] * Ω[0] * Ω[2] + tau[1] * inertia_inv[1],
        inertia_diff[2] * Ω[0] * Ω[1] + tau[2] * inertia_inv[2]
    ])

    return np.concatenate((Q_dot, Ω_dot))