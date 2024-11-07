import numpy as np

def normalize_quaternion(q):
    """Normalize a quaternion and force scalar positive.

    Args:
        q (np.ndarray): quaternion of form [q1, q2, q3, q4]
    """
    q_normalized =  q / np.linalg.norm(q)
    # print(q_normalized, -q_normalized if q_normalized[0] < 0 else q_normalized)
    #  if q_normalized[0] < 0: q_normalized *= -1
    return q_normalized

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.

    Parameters:
    q1 (np.array): First quaternion (4D vector).
    q2 (np.array): Second quaternion (4D vector).

    Returns:
    np.array: The product of the two quaternions.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Compute the product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_inverse(q):
    """
    Copute the inverse of a quaternion

    Parameters:
    qq (np.array): Quaternion (4D vector).

    Returns:
    np.array: The inverse of the quaternion

    Note:
    q must be noramliszed
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


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

def quaternion_dynamics(x, dt, inertia, tau, noise=0):
    """Apply quaternion dynamics and euler's equations to state vector x.

    NOTE: inertia tensor must be diagonal, and torque must be in the principal axis frame.

    Args:
        x (np.ndarray): state vector, of form [q1, q2, q3, q4, Ω1, Ω2, Ω3]
        dt (float): interval of time
        inertia (np.ndarray): inertia tensor
        tau (np.ndarray): torque vector in body grame
    """

    # Add random noise
    # noise_torque = np.random.normal(np.zeros(3), noise)
    # tau += noise_torque

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


def angle_axis_between(q1, q2):
    """
    Calculate the angle-axis representation of the rotation from quaternion q1 to quaternion q2.

    This function computes the angle of rotation and the axis of rotation that describes
    the transformation from one quaternion to another. The quaternions should be normalized.

    Parameters:
    q1 (np.ndarray): A 4-element array representing the first quaternion (starting orientation).
    q2 (np.ndarray): A 4-element array representing the second quaternion (target orientation).

    Returns:
    tuple: A tuple containing:
        - theta (float): The angle of rotation in radians.
        - rotation_vector (np.ndarray): A 3-element array representing the axis of rotation.
    """
    q_1_to_2 = quaternion_multiply(quaternion_inverse(q1), q2)
    q_1_to_2 = normalize_quaternion(q_1_to_2)

    theta = 2 * np.arccos(q_1_to_2[0])

    # Wrap to be between 0 and pi
    if theta > np.pi:
        theta = theta - 2 * np.pi

    if np.abs(theta) < 0.00001:
        return theta, np.zeros(3)
    
    rotation_vector = q_1_to_2[1:] / np.sin(theta / 2)

    # TODO - not sure if this works!
    # if np.abs(theta - np.pi) < 0.1:
    #     rotation_vector = np.array([1, 0, 0])

    return theta, rotation_vector


def compute_control_torque(x, x_desired, K_p=1, K_d=1, tau_max=None):
    """
    Compute torque from current to target state using P-D control.
    Can also specify a maximum actuator torque, `tau_max`, to limit
    the norm of the appliec torque
    """
    # x = [q, w]
    # x_d = [q_d, w_d]
    
    q = normalize_quaternion(x[:4])
    q_d = normalize_quaternion(x_desired[:4])
    theta, rotation_vector = angle_axis_between(q, q_d)
    
    w = x[4:]
    w_d = x_desired[4:]
    w_error = w_d - w

    tau = K_p * theta * rotation_vector + K_d * w_error

    if tau_max is not None:
        if np.linalg.norm(tau) > tau_max:
            tau = tau_max * tau / np.linalg.norm(tau) 

    return tau
