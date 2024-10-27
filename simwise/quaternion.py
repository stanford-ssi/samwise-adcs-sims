import numpy as np

def normalize_quaternion(q):
    """Normalize a quaternion.

    Args:
        q (np.ndarray): quaternion of form [q1, q2, q3, q4]
    """
    return q / np.linalg.norm(q)

def quaternion2euler(q, sequence="zyx"):
    """Transform a quaternion to euler angles.

    NOTE: only supports zyx sequence.

    Args:
        q (np.ndarray): quaternion of form [q1, q2, q3, q4]
        sequence (str): sequence of euler angles, default is "zyx"
    """
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    if sequence == "zyx":
        phi = np.arctan2((2*(q1*q2 + q3*q4)), 1 - 2*(q2**2 + q3**2))
        theta = np.arcsin(2*(q1*q3 - q4*q2))
        psi = np.arctan2(2*(q1*q4 + q2*q3), 1 - 2*(q3**2 + q4**2))
        return np.array([phi, theta, psi])
    else:
        raise ValueError("Invalid sequence")

def quaternion_dynamics(x, dt, inertia, tau):
    """Apply quaternion dynamics and euler's equations to state vector x.

    NOTE: inertia tensor must be diagonal, and torque must be in the principal axis frame.

    Args:
        x (np.ndarray): state vector, of form [q1, q2, q3, q4, Ω1, Ω2, Ω3]
        dt (float): interval of time
        inertia (np.ndarray): inertia tensor
        tau (np.ndarray): torque vector in body grame
    """
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
