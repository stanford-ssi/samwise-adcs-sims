import numpy as np 
from simwise.forces.area_projection import create_rotation_matrix, define_satellite_vertices, project_prism, calculate_projected_area
from simwise.constants import *
from simwise.data_structures.parameters import Parameters
from simwise.world_model.atmosphere_data import *
import unittest

# DRAG CALCULATION:

def dragPertubationTorque(params, e_angles, velocity, altitude):
    """
    Calculate the drag perturbation torque on the satellite.

    Args:
        params (Parameters): The parameters object containing satellite properties.
        e_angles (np.ndarray): Euler angles [psi, theta, phi], shape (3,)
        velocity (np.ndarray): Satellite velocity vector in body frame, shape (3,)
        altitude (float): altitude of satellite

    Returns:
        np.ndarray: Drag perturbation torque, shape (3,3)
            The cols represent the x,y,z value of the Drag vector
            The rows represent each of the following Drag vector values:
                1. Low Solar Activity
                2. Medium Solar Activity
                3. High Solar Activity
    """
    
    # Calculate atmospheric density from satellite altitude
    atmospheric_density = np.array([
        low_solar_activity[round(altitude)]['density'],
        moderate_solar_activity[round(altitude)]['density'],
        high_solar_activity[round(altitude)]['density']
    ])
    
    # Calculate rotation matrix
    R = create_rotation_matrix(e_angles[0], e_angles[1], e_angles[2])
    
    # Get satellite vertices
    vertices = define_satellite_vertices(params)
    
    # Rotate and project the vertices
    rotated_vertices, projected_vertices = project_prism(vertices, e_angles[0], e_angles[1], e_angles[2])
    
    # Calculate projected area
    projected_area = calculate_projected_area(projected_vertices)
    
    # Calculate drag force magnitude
    v_mag = np.linalg.norm(velocity)
    drag_force_mag = 0.5 * atmospheric_density * v_mag**2 * projected_area * drag_coefficient()
    
    # Calculate drag force vector (opposite to velocity direction)
    drag_force = -np.outer(drag_force_mag, velocity / v_mag)
    
    # Calculate torque
    r = params.Cp - params.Cg  # Vector from center of gravity to center of pressure
    torque = np.cross(r, drag_force)
    
    return torque

def drag_coefficient():
    Cd = 2.0
    return Cd
