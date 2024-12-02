import numpy as np
from simwise.math.quaternion import normalize_quaternion

def attitude_dynamics(x, dt, inertia, tau, noise=0):
    """Apply quaternion dynamics and euler's equations to state vector x.

    NOTE: inertia tensor must be diagonal, and torque must be in the principal axis frame.

    Args:
        x (np.ndarray): state vector, of form [q1, q2, q3, q4, Ω1, Ω2, Ω3]
        dt (float): interval of time
        inertia (np.ndarray): inertia tensor
        tau (np.ndarray): torque vector in body grame
    """

    # Add random noise
    noise_torque = np.random.normal(np.zeros(3), noise)
    tau += noise_torque

    Q = normalize_quaternion(x[:4])
    Ω = x[4:]
    stm = np.array([
        [0,   -Ω[0], -Ω[1], -Ω[2]],
        [Ω[0],    0,  Ω[2], -Ω[1]],
        [Ω[1],-Ω[2],     0,  Ω[0]],
        [Ω[2], Ω[1], -Ω[0],     0]
    ])
    Q_dot = 0.5 * stm @ Q
    
    Ω_dot = np.array([
        (inertia[1] - inertia[2])/inertia[0] * Ω[1] * Ω[2] + tau[0] / inertia[0],
        (inertia[2] - inertia[0])/inertia[1] * Ω[0] * Ω[2] + tau[1] / inertia[1],
        (inertia[0] - inertia[1])/inertia[2] * Ω[0] * Ω[1] + tau[2] / inertia[2]
    ])
    return np.concatenate((Q_dot, Ω_dot))