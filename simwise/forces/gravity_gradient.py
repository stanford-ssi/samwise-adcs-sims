import numpy as np
from simwise.constants import MU_EARTH
from simwise.math.quaternion import rotate_vector_by_quaternion

"""useful video --> https://www.youtube.com/watch?v=UqUPDtu2s0M"""

def vector_to_quaternion(v):
    """
    Convert vector to pure quaternion (scalar part = 0)
    Args:
        v (np.ndarray): 3D vector
    Returns:
        np.ndarray: Pure quaternion [0, v1, v2, v3]
    """
    v = np.array(v)
    return np.array([0.0, v[0], v[1], v[2]])

def gravity_gradient_perturbation_torque(q, r_vector, inertia):
    """
    Calculate the gravity gradient perturbation torque on the satellite using quaternions.
    Args:
        q (np.ndarray): Quaternion [q0, q1, q2, q3], shape (4,)
        r_vector (np.ndarray): Position vector in ECI frame, shape (3,)
        inertia (np.ndarray): Principal moments of inertia, shape (3,)
    Returns:
        np.ndarray: Gravity gradient perturbation torque, shape (3,)
    """
    # Transform r_vec to body frame
    r_body = rotate_vector_by_quaternion(r_vector, q)

    r_mag = np.linalg.norm(r_body) 
    r_hat = r_body / r_mag

    coeff = 3 * MU_EARTH / r_mag ** 3

    I_body = np.diag(inertia)

    torque = coeff * np.cross(r_hat, np.dot(I_body, r_hat))
    
    return torque