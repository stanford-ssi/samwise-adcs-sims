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

def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate a vector from inertial to body frame using quaternion.
    """
    q = normalize_quaternion(q)
    v_quat = np.array([0, v[0], v[1], v[2]])
    q_conj = quaternion_inverse(q)
    temp = quaternion_multiply(q, v_quat)
    v_rotated = quaternion_multiply(temp, q_conj)
    return v_rotated[1:]
