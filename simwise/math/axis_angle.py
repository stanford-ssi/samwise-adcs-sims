import numpy as np

def vectors_to_axis_angle(v1, v2):
    """
    Calculates the axis and angle of rotation between two vectors.
    """

    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Calculate dot product and cross product
    dot_product = np.dot(v1_norm, v2_norm)
    cross_product = np.cross(v1_norm, v2_norm)

    # Calculate angle
    angle = np.arccos(np.clip(dot_product, -1, 1))

    # Calculate axis
    axis = cross_product / np.linalg.norm(cross_product)

    return axis, angle