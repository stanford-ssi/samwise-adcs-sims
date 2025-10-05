"""
Axis-Angle specific operations
"""
import numpy as np

from simwise.math.unit_conversions import RAD_TO_DEG

def axis_angle_between_vectors(v1: np.ndarray, v2: np.ndarray, degrees = False) -> tuple[float, np.ndarray]:
    """Calculates the axis and angle of rotation between two vectors.
    Represents an active transformation from vector 1 to vector 2.

    Args:
        v1 (np.ndarray): vector 1
        v2 (np.ndarray): vector 2
        degrees (bool, optional): Units to express resultant angle. Defaults to False.

    Returns:
        tuple[float, np.ndarray]: axis, angle
    """

    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Calculate angle
    dot_product_normalized = np.dot(v1_norm, v2_norm)
    angle = np.arccos(np.clip(dot_product_normalized, -1, 1))

    if angle == 0:
        return 0, np.zeros(3)

    # Calculate axis
    cross_product = np.cross(v1, v2)
    axis = cross_product / np.linalg.norm(cross_product)

    if degrees:
        angle *= RAD_TO_DEG

    return float(angle), axis