import numpy as np
from simwise.math.quaternion import dcm_to_quaternion

def compute_nadir_pointing(r_eci):
    """
    This function computes the control torque needed to point the satellite towards the sun while maintaining nadir pointing.
    
    Inputs:
        r_sun_eci: np.ndarray   -   This is the sun position in the ECI frame, vector points from Earth to Sun
        r_eci: np.ndarray       -   This is the satellite position in the ECI frame, vector points from Earth to Satellite
    """
    
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
    
    return target_attitude
    