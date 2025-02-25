import numpy as np
from simwise.math.area_projection import create_rotation_matrix, define_satellite_vertices, project_prism, calculate_projected_area
from simwise.constants import *
from simwise.world_model.atmosphere import *


def dragPertubationTorque(params, r_eci, e_angles, velocity, atmospheric_density):
    """
    Calculate the drag perturbation torque on the satellite.

    Args:
        params (Parameters): The parameters object containing satellite properties.
        r_eci (np.ndarray): Position vector in ECI coordinate frame
        e_angles (np.ndarray): Euler angles [psi, theta, phi], shape (3,)
        velocity (np.ndarray): Satellite orbital velocity vector in body frame, shape (3,)
        altitude (float): altitude of satellite

    Returns:
        np.ndarray: Drag perturbation torque, shape (3,3)
            The cols represent the x,y,z value of the Drag vector
            The rows represent each of the following Drag vector values:
                1. Low Solar Activity
                2. Medium Solar Activity
                3. High Solar Activity
    """
    
    # Calculate rotation matrix
    R = create_rotation_matrix(e_angles[0], e_angles[1], e_angles[2])
    
    # Get satellite vertices
    vertices = define_satellite_vertices(params)
    
    # Rotate and project the vertices
    rotated_vertices, projected_vertices = project_prism(vertices, e_angles[0], e_angles[1], e_angles[2])
    
    # Calculate projected area
    projected_area = calculate_projected_area(projected_vertices)
    
    # Calculate velocity relative to air
    relative_velocity = find_relative_air_velocity(r_eci, velocity)

    # Calculate drag force magnitude
    v_mag = np.linalg.norm(relative_velocity)
    drag_force_mag = 0.5 * atmospheric_density * v_mag**2 * projected_area * SAT_CD
    
    # Calculate drag force vector (opposite to velocity direction)
    drag_force = -np.outer(drag_force_mag, velocity / v_mag)
    
    # Calculate torque
    r = params.Cp - params.Cg  # Vector from center of gravity to center of pressure
    torque = np.cross(r, drag_force.flatten())
    
    return torque

def find_air_velocity(r_eci):
    """
    Calculate the relative velocity of the satellite to the air molecules,
    assuming that the air molecules move with the angular velocity of Earth.
    Parameters:
    r_eci: current orbital position of satellite in ECI frame (km)
    Return:
    v_wind: Velocity of air particles at this position in ECI frame (m/s), assuming they move with same angular velocity as Earth (first approximation)
    """
    # Define rotational velocity vector (rad/s)
    w = 2*np.pi/SECONDS_PER_DAY
    w_vec = np.array([0, 0, w])
    
    # Debug: Print shapes
    # print(f"w_vec shape: {w_vec.shape}")
    # print(f"r_eci shape: {np.array(r_eci).shape}")
    # print(f"r_eci value: {r_eci}")
    
    # Make sure r_eci is a 3D vector
    r_eci_3d = np.array(r_eci)
    if len(r_eci_3d.shape) == 1 and r_eci_3d.shape[0] == 3:
        # Already a 3D vector, good to go
        r_eci_vec = r_eci_3d
    else:
        # Try to handle other cases or raise a more specific error
        raise ValueError(f"r_eci must be a 3D vector, got shape {r_eci_3d.shape}")
    
    # v = w x r
    v_wind = np.cross(w_vec, r_eci_vec*1000)
    return v_wind

def find_relative_air_velocity(r_eci, v):
    """
    Calculate the relative velocity of the satellite to the air molecules,
    assuming that the air molecules move with the angular velocity of Earth.

    Parameters:
    r_eci: current orbital position of satellite in ECI frame (km)
    v: current orbital velocity in ECI frame (m/s)

    Return: 
    v_rel: relative velocity between satellite and air particles (m/s)
    """
    # Get air velocity in km/s
    v_wind = find_air_velocity(r_eci)
    
    # Calculate relative velocity (v_satellite - v_air)
    v_rel = np.array(v) - v_wind
    
    return v_rel




# import numpy as np
# from simwise.data_structures.parameters import Parameters

# def init():
#     # Create a Parameters object
#     params = Parameters()
    
#     # Set up test cases
#     test_cases = [
#         {
#             "name": "Low altitude, low velocity",
#             "e_angles": np.array([0, 0, 0]),
#             "velocity": np.array([1000, 0, 0]),
#             "altitude": 200000  # 200 km
#         },
#         {
#             "name": "Medium altitude, medium velocity",
#             "e_angles": np.array([np.pi/2, 0, 0]),
#             "velocity": np.array([7600, 0, 0]),
#             "altitude": 450000  # 450 km
#         },
#         {
#             "name": "High altitude, high velocity",
#             "e_angles": np.array([np.pi/2, np.pi/2, np.pi/2]),
#             "velocity": np.array([0, 0, 8000]),
#             "altitude": 800000  # 800 km
#         }
#     ]
    
#     # Run test cases
#     for case in test_cases:
#         print(f"\nTest Case: {case['name']}")
#         torque = dragPertubationTorque(params, case['e_angles'], case['velocity'], case['altitude'])
        
#         print(f"Euler Angles: {case['e_angles']}")
#         print(f"Velocity: {case['velocity']} m/s")
#         print(f"Altitude: {case['altitude']} m")
#         print("Drag Perturbation Torque:")
#         print("  Low Solar Activity:    ", torque[0])
#         print("  Low Solar Activity - MAGNITUDE:    ", np.linalg.norm(torque[0]))
#         print("  Medium Solar Activity: ", torque[1])
#         print("  Medium Solar Activity - MAGNITUDE:    ", np.linalg.norm(torque[1]))
#         print("  High Solar Activity:   ", torque[2])
#         print("  High Solar Activity - MAGNITUDE:    ", np.linalg.norm(torque[2]))
        

# if __name__ == "__main__":
#     init()