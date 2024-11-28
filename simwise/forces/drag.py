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
    
    # Convert altitude to nearest multiple of 20 km
    altitude_km = round(altitude / 1000 / 20) * 20

    # Get the min and max altitudes from the data
    min_altitude = min(low_solar_activity.keys())
    max_altitude = max(low_solar_activity.keys())

    # Clamp the altitude to the valid range
    altitude_km = max(min_altitude, min(max_altitude, altitude_km))

    atmospheric_density = np.array([
        low_solar_activity[altitude_km]['density'],
        moderate_solar_activity[altitude_km]['density'],
        high_solar_activity[altitude_km]['density']
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


import numpy as np
from simwise.data_structures.parameters import Parameters

def init():
    # Create a Parameters object
    params = Parameters()
    
    # Set up test cases
    test_cases = [
        {
            "name": "Low altitude, low velocity",
            "e_angles": np.array([0, 0, 0]),
            "velocity": np.array([1000, 0, 0]),
            "altitude": 200000  # 200 km
        },
        {
            "name": "Medium altitude, medium velocity",
            "e_angles": np.array([np.pi/2, 0, 0]),
            "velocity": np.array([7600, 0, 0]),
            "altitude": 450000  # 450 km
        },
        {
            "name": "High altitude, high velocity",
            "e_angles": np.array([np.pi/2, np.pi/2, np.pi/2]),
            "velocity": np.array([0, 0, 8000]),
            "altitude": 800000  # 800 km
        }
    ]
    
    # Run test cases
    for case in test_cases:
        print(f"\nTest Case: {case['name']}")
        torque = dragPertubationTorque(params, case['e_angles'], case['velocity'], case['altitude'])
        
        print(f"Euler Angles: {case['e_angles']}")
        print(f"Velocity: {case['velocity']} m/s")
        print(f"Altitude: {case['altitude']} m")
        print("Drag Perturbation Torque:")
        print("  Low Solar Activity:    ", torque[0])
        print("  Low Solar Activity - MAGNITUDE:    ", np.linalg.norm(torque[0]))
        print("  Medium Solar Activity: ", torque[1])
        print("  Medium Solar Activity - MAGNITUDE:    ", np.linalg.norm(torque[1]))
        print("  High Solar Activity:   ", torque[2])
        print("  High Solar Activity - MAGNITUDE:    ", np.linalg.norm(torque[2]))
        

if __name__ == "__main__":
    init()