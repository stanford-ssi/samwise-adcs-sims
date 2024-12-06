import numpy as np

from simwise.math.quaternion import dcm_to_quaternion

def compute_sun_pointing_nadir_constrained(r_sun_eci, r_eci):
    """
    This function computes the control torque needed to point the satellite towards the sun while maintaining nadir pointing.
    """
    # Get the sun position in the ECI frame
    # TODO: replace this with sun position (not normalized) and subtract
    # current position
    # return np.array([1.0, 0.0, 0.0, 0.0])
    sun_eci = -np.array(r_sun_eci) / np.linalg.norm(r_sun_eci)
    
    # Get the nadir vector in the ECI frame
    nadir_eci = -np.array(r_eci) / np.linalg.norm(r_eci)

    
    
    # Project nadir vector onto sun vector and find the component of the nadir vector that is perpendicular to the sun vector
    nadir_perpendicular = nadir_eci - np.dot(nadir_eci, sun_eci) * sun_eci
    norm_perpendicular = np.linalg.norm(nadir_perpendicular)

    
    
    if norm_perpendicular < 1e-6:
        raise ValueError("Nadir vector is nearly collinear with sun vector.")
    nadir_perpendicular = nadir_perpendicular / norm_perpendicular

    # Get the cross product of the sun and nadir vectors in the body frame
    third = np.cross(sun_eci, nadir_perpendicular)
    third = third / np.linalg.norm(third)
    
    # Get the rotation matrix from the ECI frame to the body frame
    # TODO double check order of vectors
    R_eci_to_sun = np.array([sun_eci, nadir_perpendicular, third]).T
    
    # Get the rotation matrix
    target_attitude = dcm_to_quaternion(R_eci_to_sun)
    
    return target_attitude
    