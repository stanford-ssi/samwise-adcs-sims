"""
Quaternion Operations
-----

References:
- https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf
"""
import numpy as np
from numpy.linalg import norm

import simwise.math.unit_conversions as conversion


# --------------------------------------------------
#   Basic
# --------------------------------------------------

def conj(q: list | np.ndarray) -> np.ndarray:
    """Finds quaternion conjugate by flipping the vector part.
    This flips the sign of the angle that the quaternion applies to a vector.

    Args:
        q (list | np.ndarray): quaternion

    Returns:
        np.ndarray: conjugate quaternion
    """

    assert len(q) == 4
    q_conj = [
        q[0],
        -q[1],
        -q[2],
        -q[3]
    ]
    return np.array(q_conj)

def inv(q: list | np.ndarray) -> np.ndarray:
    """Generates quaternion inverse

    Args:
        q (list | np.ndarray): quaternion

    Returns:
        np.ndarray: quaternion inverse
    """
    q_conj = conj(q)
    return q_conj / (norm(q) ** 2)

def unit(q: list | np.ndarray, tol = 0.000001):
    """Unit quaternion. If norm is smaller than tolerance, then sets output quaternion to (4,) zeros

    Args:
        q (list | np.ndarray): Quaternion to be normalized (unit-ized)
        tol (float, optional): tolerance for zero vector (to prevent weird singularity behavior). Defaults to 0.000001.

    Returns:
        np.ndarray: unit quaternion
    """

    q = np.array(q)
    if abs(norm(q)) < tol:
        return np.zeros(len(q))
    
    return q / norm(q)

# --------------------------------------------------
#   Angle Axis
# --------------------------------------------------

#----- Quaternion and Axis rotation -----#
def angle_axis_to_q(angle: float, axis: list | np.ndarray, degrees = False) -> np.ndarray:
    """Turn angle-axis rotation into quaternion

    Args:
        angle (float): angle
        axis (list | np.ndarray): axis
        degrees (bool, optional): unit of angle parameter. Defaults to False.

    Returns:
        np.ndarray: quaternion
    """
    if isinstance(axis, list):
        axis = np.array(axis)
    angle_rad = (angle 
                 if not degrees 
                 else angle * conversion.DEG_TO_RAD)
    axis_norm = axis / norm(axis)

    w = np.cos(angle_rad/2)
    x,y,z = np.sin(angle_rad / 2) * axis_norm

    return np.array([w,x,y,z])

def q_to_axis_angle(quat: list | np.ndarray, degrees = False) -> tuple[float, np.ndarray]:
    """Turns quaternion into angle-axis

    Args:
        quat (list | np.ndarray): quaternion
        degrees (bool, optional): units of returned angle. Defaults to False.

    Returns:
        (float, np.ndarray): tuple with angle and axis, where angle is expressed in units specified, and axis is a 3-dimensional np array
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    
    angle = 2*np.arccos(w)

    if angle == 0:
        return 0, np.zeros(3)
    
    i = x/np.sin(angle/2)
    j = y/np.sin(angle/2)
    k = z/np.sin(angle/2)
    angle = (angle 
             if not degrees 
             else angle * conversion.RAD_TO_DEG)
    return angle, np.array([i, j, k])

# --------------------------------------------------
#   Algebra
# --------------------------------------------------

def multiply(q1: list | np.ndarray, q2: list | np.ndarray):
    """Multiplies quaternions. Inputs must be size 4 numpy arrays

    Args:
        q1 (list | np.ndarray): quaternion 1
        q2 (list | np.ndarray): quaternion 2

    Returns:
        _type_: multiplied quaternion
    """
    p0, p1, p2, p3 = q1
    q0, q1, q2, q3 = q2

    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0
    ])

def apply(quat: np.ndarray, vector: np.ndarray, rotation_type = "passive") -> np.ndarray:
    """Equivalent to a rotation of a vector. 

    Types of rotations specified by the argument `rotation_type`:
    - "active" (default) - Rotates a vector V around a FIXED axis to product V'
    - "passive" - Rotates the coordinate frame, and produces the vector representation in that new frame

    Args:
        quat (np.ndarray): _description_
        vector (np.ndarray): _description_
        rotation_type (str, optional): "active" or "passive". Defaults to "passive".

    Returns:
        np.ndarray: For active rotation, the new vector. 
        For passive rotation, the vector expressed in the new coordinate system.
    """

    # Passive rotation is "equivalent" to rotating the other way
    if rotation_type == "passive":
        quat = conj(quat)

    # Normalizes quaternion
    quat = unit(quat)

    temp = multiply(quat, [0, *list(vector)])
    rslt = multiply(temp, conj(quat))

    if abs(rslt[0]) > 0.0001:
        print(f"[DEBUG] Quanternion is not normalized. Result vector of {rslt}") 
    
    # Discards 
    return rslt[1:]