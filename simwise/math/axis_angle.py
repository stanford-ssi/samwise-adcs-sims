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
    
    # Handle parallel/antiparallel cases
    cross_norm = np.linalg.norm(cross_product)
    if cross_norm < 1e-10:  # Vectors are parallel or antiparallel
        # For parallel vectors (angle ≈ 0), any perpendicular axis works
        # We'll return [0, 0, 1] if v1 is not parallel to it, otherwise [0, 1, 0]
        if angle < 1e-10:
            if abs(np.dot(v1_norm, [0, 0, 1])) < 0.9:
                return np.array([0, 0, 1]), 0.0
            else:
                return np.array([0, 1, 0]), 0.0
        # For antiparallel vectors (angle ≈ π), similar approach
        else:
            if abs(np.dot(v1_norm, [0, 0, 1])) < 0.9:
                return np.array([0, 0, 1]), np.pi
            else:
                return np.array([0, 1, 0]), np.pi
    
    # Normal case - normalize the cross product
    axis = cross_product / cross_norm
    return axis, angle