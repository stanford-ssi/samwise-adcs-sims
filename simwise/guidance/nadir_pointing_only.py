import numpy as np
from simwise.math.quaternion import dcm_to_quaternion, ensure_quaternion_continuity

# Debug tracker
nadir_pointing_calls = []  # Will store 1s when function is called

# Variable to store the previous quaternion for continuity checks
_previous_quaternion = None

def compute_nadir_pointing(r_eci):
    """
    Computes the target attitude quaternion for nadir pointing and the nadir vector.
    
    This function calculates the quaternion that orients the satellite's body frame
    such that the z-axis points toward Earth (nadir direction).
    
    Args:
        r_eci (np.ndarray): Satellite position in ECI frame [x, y, z] in meters
        
    Returns:
        tuple: A tuple containing:
            - target_attitude (np.ndarray): Quaternion [q0, q1, q2, q3] representing the desired attitude
            - nadir_eci (np.ndarray): Unit vector [x, y, z] pointing from satellite to Earth center (nadir)
    """
    global _previous_quaternion
    
    # Debug: Record that function was called
    nadir_pointing_calls.append(1)
    
    # Get the nadir vector in the ECI frame
    # We normalize to get a unit direction vector
    # The negative sign makes this point from the satellite to the earth
    # This is also known as the nadir vector
    nadir_eci = -np.array(r_eci) / np.linalg.norm(r_eci)
    
    # Choose a reference vector that's likely not parallel to nadir
    # We'll use the ECI z-axis as an initial guess
    ref = np.array([0, 0, 1])
    # It does not matter what we choose here, as long as it is not parallel to nadir_eci
    
    # Make sure ref is not parallel to nadir_eci
    if np.abs(np.dot(ref, nadir_eci)) > 0.9:
        # If it IS parallel, then use the ECI x-axis as a reference
        # Either vector will work because you can't be parallel to both
        ref = np.array([1, 0, 0])
    
    # Compute y-axis perpendicular to nadir and reference
    y_body = np.cross(nadir_eci, ref)
    y_body = y_body / np.linalg.norm(y_body)
    
    # Compute x-axis to complete right-handed system
    x_body = np.cross(y_body, nadir_eci)
    x_body = x_body / np.linalg.norm(x_body)
    
    # The rotation matrix from ECI to body frame
    # z-axis points to nadir, x and y are in the orbital plane
    R_eci_to_body = np.array([x_body, y_body, nadir_eci]).T
    
    # Convert rotation matrix to quaternion
    target_attitude = dcm_to_quaternion(R_eci_to_body)
    
    # Apply quaternion continuity check if we have a previous quaternion
    if _previous_quaternion is not None:
        # If dot product is negative, flip the sign of the current quaternion
        dot_product = np.sum(_previous_quaternion * target_attitude)
        if dot_product < 0:
            target_attitude = -target_attitude
    
    # Update the previous quaternion for next call
    _previous_quaternion = target_attitude.copy()
    
    return target_attitude, nadir_eci
    