import numpy as np
from simwise.math.quaternion import rotate_vector_by_quaternion

# Define photodiode normals for cube faces

def sun_in_body_frame(v_sun_eci, q):
    """Generate a measurement for the current state"""
    # Sun sensor measurement
    return rotate_vector_by_quaternion(v_sun_eci, q)

def generate_photodiode_measurements(r_sun_body, photodiode_normals):
    # TODO interpolate the datasheet curve
    half_angle = np.radians(70)  # Convert to radians
    max_response = 3.3          # Maximum voltage
    
    measurements = []
    for normal in photodiode_normals:
        # Calculate cosine of angle between sun vector and photodiode normal
        cos_theta = np.dot(normal, r_sun_body)
        
        # If the sun is behind the photodiode, the response is zero
        # TODO add earth albedo
        if cos_theta <= 0:
            response = 0
        else:
            response = cos_theta
        
        # Map to voltage
        voltage = response * max_response
        measurements.append(voltage)
        
    return np.array(measurements)


def sun_vector_ospf(measurements):
    # Convert measurements to unit vectors
    measurements = measurements / np.linalg.norm(measurements)
    
    # Calculate the sun vector
    sun_vector = np.zeros(3)
    sun_vector = np.array([
        measurements[0] - measurements[1], # x+ - x-
        measurements[2] - measurements[3], # y+ - y-
        measurements[4] - measurements[5]  # z+ - z-
    ])
    
    return sun_vector / np.linalg.norm(sun_vector)

def sun_vector_pyramid(y_m, y_p, z_m, z_p):
    psi = np.arctan2(y_m, y_p)
    theta = np.arctan2(z_m, z_p)
    
    return psi, theta
    