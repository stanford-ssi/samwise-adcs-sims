import numpy as np

from simwise.math.quaternion import dcm_to_quaternion, regularize_quaternion, normalize_quaternion

def compute_sun_pointing_nadir_constrained(v_sun_eci, r_eci):
    """
    This function computes the control torque needed to point the satellite towards the sun while maintaining nadir pointing.
    """
    # Get the sun position in the ECI frame
    # TODO: replace this with sun position (not normalized) and subtract
    # current position
    # return np.array([1.0, 0.0, 0.0, 0.0])
    sun_eci = -np.array(v_sun_eci) / np.linalg.norm(v_sun_eci)
    
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

def compute_nadir_pointing_velocity_constrained(r_eci, v_eci, orbit_period):
    """
    This function computes the control torque needed to point the satellite towards the sun while maintaining nadir pointing.
    """
    # Get the nadir vector in the ECI frame
    radial = np.array(r_eci) / np.linalg.norm(r_eci)
    radial = radial / np.linalg.norm(radial)
    normal = np.cross(r_eci, v_eci)
    normal = normal / np.linalg.norm(normal)
    transverse = np.cross(normal, radial)
    
    # Get the rotation matrix from the ECI frame to the body frame
    # TODO double check order of vectors
    R_target = np.array([radial, transverse, normal]).T
    
    # Get the rotation matrix
    target_attitude = dcm_to_quaternion(R_target)
    target_attitude = regularize_quaternion(target_attitude)
    target_attitude = normalize_quaternion(target_attitude)
    
    # Compute the target rotation rates
    target_omega = np.array([0.0, 0.0, 0.0])
    target_omega[2] = 2 * np.pi / orbit_period
    
    return target_attitude, target_omega
    