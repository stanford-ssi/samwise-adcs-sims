import numpy as np 
import simwise.constants
from scipy.spatial import ConvexHull

def dragPertubationTorque(Cpa, Cg, e_angles):
    """ Calculate the drag pertubation torque

    Args:
        elements (np.ndarray): orbital elements of form [a, e, i, Ω, ω, θ]
    """
    
    
def calculateWettedArea(e_angles):
    # Given Euler angles, figure out the orientation of the satellite wrt the velocity vector
    pass

def create_rotation_matrix(psi, theta, phi):
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    return Rz @ Ry @ Rx

def project_prism(l, w, h, psi, theta, phi):
    # Define vertices of the prism with the correct orientation
    vertices = np.array([
        [0, 0, 0], [0, h, 0], [0, h, w], [0, 0, w],
        [l, 0, 0], [l, h, 0], [l, h, w], [l, 0, w]
    ])

    # Create rotation matrix
    R = create_rotation_matrix(psi, theta, phi)

    # Rotate vertices
    rotated_vertices = np.dot(vertices, R.T)

    # Project onto viewing plane (yz-plane since viewing direction is along x-axis)
    projected_vertices = rotated_vertices[:, 1:]  # y and z axes

    return projected_vertices

def calculate_projected_area(projected_vertices):
    # Use shoelace formula to calculate the area
    n = len(projected_vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += projected_vertices[i][0] * projected_vertices[j][1]
        area -= projected_vertices[j][0] * projected_vertices[i][1]
    return abs(area) / 2.0

# Main calculation
l = 0.1 * np.sqrt(2)  # Length along x-axis
w = 0.1  # Width along z-axis
h = 0.54  # Height along y-axis

# Example rotation angles (in radians)
psi = np.pi / 4    # 45 degrees
theta = np.pi / 6  # 30 degrees
phi = np.pi / 3    # 60 degrees

projected_vertices = project_prism(l, w, h, psi, theta, phi)
projected_area = calculate_projected_area(projected_vertices)

print(f"Projected vertices:\n{projected_vertices}")
print(f"\nProjected area: {projected_area:.6f} square meters")