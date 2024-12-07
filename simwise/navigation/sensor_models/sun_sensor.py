import numpy as np
from simwise.math.quaternion import rotate_vector_by_quaternion

# Define photodiode normals for cube faces

def sun_in_body_frame(r_sun_eci, q):
    """Generate a measurement for the current state"""
    # Sun sensor measurement
    return rotate_vector_by_quaternion(r_sun_eci, q)

def generate_photodiode_measurements(r_sun_body, photodiode_normals):
    half_angle = np.radians(70)  # Convert to radians
    max_response = 3.3          # Maximum voltage
    
    measurements = []
    for normal in normals:
        # Calculate cosine of angle between sun vector and photodiode normal
        cos_theta = np.dot(normal, r_sun_body)
        
        # Ensure the value is within valid range
        cos_theta = np.clip(cos_theta, -1, 1)
        
        # Convert to angle
        theta = np.arccos(cos_theta)
        
        # Calculate response
        response = np.sin(theta)
        # if theta <= half_angle:
        #     response = np.sin(theta)
        # else:
        #     response = 0
        
        # Map to voltage
        voltage = response * max_response
        measurements.append(voltage)
    
    return np.array(measurements)


