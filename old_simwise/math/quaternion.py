import numpy as np
from numba import jit

@jit(nopython=True)
def normalize_quaternion(q):
    """Normalize a quaternion and force scalar positive.

    Args:
        q (np.ndarray): quaternion of form [q1, q2, q3, q4]
    """
    q_normalized =  q / np.linalg.norm(q)
    # print(q_normalized, -q_normalized if q_normalized[0] < 0 else q_normalized)
    #  if q_normalized[0] < 0: q_normalized *= -1
    return q_normalized

import numpy as np

@jit(nopython=True)
def regularize_quaternion(q):
    """
    Regularize a quaternion so that it has a consistent 'direction'.
    The convention here is:
    1. q_w (the first element) should be non-negative.
    2. If q_w = 0, then we apply a fallback rule by checking subsequent elements.

    This is useful for testing that two quaternions represent the same rotation.

    Parameters:
    q (np.ndarray): A quaternion array [q_w, q_x, q_y, q_z].
    
    Returns:
    np.ndarray: The regularized quaternion.
    """
    # Make a copy to avoid modifying in-place
    _q = q
    q = np.ones(4)
    q[1] = _q[1]
    q[2] = _q[2]
    q[3] = _q[3]
    q[0] = _q[0]

    # If q_w is positive, we are good
    if q[0] > 0.0:
        return q

    # If q_w is zero or negative, we need to decide how to handle:
    if q[0] < 0.0:
        # Just flip all signs if q_w < 0
        q = -q
    else:
        # q_w == 0
        # Fallback rule: ensure the first non-zero component among (q_x, q_y, q_z)
        # is positive. For example:
        if q[1] < 0.0:
            q = -q
        elif q[1] == 0.0:
            if q[2] < 0.0:
                q = -q
            elif q[2] == 0.0:
                # If q_y is also zero, check q_z
                if q[3] < 0.0:
                    q = -q
                # If q_z also zero (unlikely for a valid unit quaternion), do nothing.

    return q


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

    return normalize_quaternion(np.array([w, x, y, z]))


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


def quaternion_to_euler(q, sequence="zyx"):
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


def euler_to_quaternion(euler_angles, sequence="zyx"):
    """Transform euler angles to a quaternion.

    NOTE: only supports zyx sequence.

    Args:
        euler_angles (np.ndarray): euler angles of form [phi, theta, psi]
        sequence (str): sequence of euler angles, default is "zyx"
    """
    phi = euler_angles[0]
    theta = euler_angles[1]
    psi = euler_angles[2]
    if sequence == "zyx":
        q1 = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
        q2 = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)
        q3 = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
        q4 = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
        return np.array([q1, q2, q3, q4])
    else:
        raise ValueError("Invalid sequence")


def quaternions_to_axis_angle(q1, q2):
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


def dcm_to_quaternion(dcm):
    """
    Convert a direction cosine matrix to a quaternion (scalar-first convention).

    Parameters:
    dcm (np.ndarray): A 3x3 direction cosine matrix.

    Returns:
    np.ndarray: The quaternion representation of the DCM.
    """
    # Extract elements from DCM
    d11, d12, d13 = dcm[0, 0], dcm[0, 1], dcm[0, 2]
    d21, d22, d23 = dcm[1, 0], dcm[1, 1], dcm[1, 2]
    d31, d32, d33 = dcm[2, 0], dcm[2, 1], dcm[2, 2]

    # Compute squared terms for each quaternion component
    q_w_sq = (1.0 + d11 + d22 + d33) / 4.0
    q_x_sq = (1.0 + d11 - d22 - d33) / 4.0
    q_y_sq = (1.0 - d11 + d22 - d33) / 4.0
    q_z_sq = (1.0 - d11 - d22 + d33) / 4.0

    # Choose the largest to ensure numerical stability
    max_index = np.argmax([q_w_sq, q_x_sq, q_y_sq, q_z_sq])

    if max_index == 0:
        # q_w is largest
        q_w = np.sqrt(q_w_sq)
        q_x = (d32 - d23) / (4.0 * q_w)
        q_y = (d13 - d31) / (4.0 * q_w)
        q_z = (d21 - d12) / (4.0 * q_w)
    elif max_index == 1:
        # q_x is largest
        q_x = np.sqrt(q_x_sq)
        q_w = (d32 - d23) / (4.0 * q_x)
        q_y = (d12 + d21) / (4.0 * q_x)
        q_z = (d13 + d31) / (4.0 * q_x)
    elif max_index == 2:
        # q_y is largest
        q_y = np.sqrt(q_y_sq)
        q_w = (d13 - d31) / (4.0 * q_y)
        q_x = (d12 + d21) / (4.0 * q_y)
        q_z = (d23 + d32) / (4.0 * q_y)
    else:
        # q_z is largest
        q_z = np.sqrt(q_z_sq)
        q_w = (d21 - d12) / (4.0 * q_z)
        q_x = (d31 + d13) / (4.0 * q_z)
        q_y = (d23 + d32) / (4.0 * q_z)

    q = np.array([q_w, q_x, q_y, q_z])

    # Regularize quaternion: enforce scalar part (w) to be non-negative
    if q[0] < 0:
        q = -q

    return q


def quaternion_to_dcm(q):
    """
    Convert a quaternion to a passive direction cosine matrix.

    Parameters:
    q (np.ndarray): A 4-element array representing the quaternion.

    Returns:
    np.ndarray: The direction cosine matrix representation of the quaternion.
    """
    q_w, q_x, q_y, q_z = q
    dcm = np.array([
        [q_w**2 + q_x**2 - q_y**2 - q_z**2, 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w), q_w**2 - q_x**2 + q_y**2 - q_z**2, 2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), q_w**2 - q_x**2 - q_y**2 + q_z**2]
    ])
    return dcm

def error_quaternion(q1, q2):
    """
    Compute the error quaternion between two quaternions.

    Parameters:
    q1 (np.ndarray): A 4-element array representing the first quaternion.
    q2 (np.ndarray): A 4-element array representing the second quaternion.

    Returns:
    np.ndarray: The error quaternion between the two quaternions.
    """
    q1_inv = quaternion_inverse(q1)
    error = quaternion_multiply(q1_inv, q2)
    return error