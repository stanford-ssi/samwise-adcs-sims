import numpy as np 
from simwise.constants import *
from scipy.spatial import ConvexHull, Delaunay
from simwise.data_structures.parameters import Parameters
from simwise.world_model.atmosphere_data import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon


# DRAG CALCULATION:

def dragPertubationTorque(Cp, Cg, e_angles, velocity, altitude):
    """
    Calculate the drag perturbation torque on the satellite.

    Args:
        Cp (np.ndarray): Center of pressure, shape (3,)
        Cg (np.ndarray): Center of gravity, shape (3,)
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
    params = Parameters()
    
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
    
    # Rotate vertices
    rotated_vertices = np.dot(vertices, R.T)
    
    # Project vertices onto plane perpendicular to velocity vector
    v_unit = velocity / np.linalg.norm(velocity)
    projected_vertices = rotated_vertices - np.outer(np.dot(rotated_vertices, v_unit), v_unit)
    
    # Calculate projected area
    projected_area = calculate_projected_area(projected_vertices[:, 1:])
    
    # Calculate drag force magnitude
    v_mag = np.linalg.norm(velocity)
    drag_force_mag = 0.5 * atmospheric_density * v_mag**2 * projected_area * drag_coefficient()
    
    # Calculate drag force vector (opposite to velocity direction)
    drag_force = -np.outer(drag_force_mag, v_unit)
    
    # Calculate torque
    r = Cp - Cg  # Vector from center of gravity to center of pressure
    torque = np.cross(r, drag_force)
    
    return torque

def drag_coefficient():
    Cd = 2.0
    return Cd
    


# AREA PROJECTION CALCULATION:
    
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

def define_satellite_vertices_simple(params):
    """
    Define the vertices of a unit cube centered at the origin.
    
    Args:
        params (Parameters): The parameters object (not used in this simple case).
    
    Returns:
        np.ndarray: Array of vertex coordinates for a unit cube centered at (0,0,0).
    """
    # Define half-length of the cube's side
    h = 0.5
    
    return np.array([
        [-h, -h, -h], [-h, -h, h], [-h, h, -h], [-h, h, h],
        [h, -h, -h], [h, -h, h], [h, h, -h], [h, h, h]
    ])
    
def define_satellite_vertices(params):
    """
    Define the vertices of the satellite based on its dimensions.
    
    Args:
        params (Parameters): The parameters object containing satellite dimensions.
    
    Returns:
        np.ndarray: Array of vertex coordinates.
    """
    #l, w, h = params.satellite_length, params.satellite_width, params.satellite_height
    l = (2 * SOLARPANEL_WIDTH + CUBESAT_WIDTH * np.sqrt(2))
    w = CUBESAT_WIDTH * np.sqrt(2)
    h = SOLARPANEL_HEIGHT
    
    return np.array([
        [CUBESAT_WIDTH/2, -CUBESAT_WIDTH/2, -CUBESAT_HEIGHT/2], [CUBESAT_WIDTH/2, CUBESAT_WIDTH/2, -CUBESAT_HEIGHT/2],          # Cube Bottom Front
        [-CUBESAT_WIDTH/2, -CUBESAT_WIDTH/2, -CUBESAT_HEIGHT/2], [-CUBESAT_WIDTH/2, CUBESAT_WIDTH/2, -CUBESAT_HEIGHT/2],        # Cube Bottom Back
        [CUBESAT_WIDTH/2, -CUBESAT_WIDTH/2, CUBESAT_HEIGHT/2], [CUBESAT_WIDTH/2, CUBESAT_WIDTH/2, CUBESAT_HEIGHT/2],            # Cube Top Front
        [-CUBESAT_WIDTH/2, -CUBESAT_WIDTH/2, CUBESAT_HEIGHT/2], [-CUBESAT_WIDTH/2, CUBESAT_WIDTH/2, CUBESAT_HEIGHT/2],          # Cube Top Back
        [CUBESAT_WIDTH/2, 0, -CUBESAT_HEIGHT/2], [CUBESAT_WIDTH/2, 0, CUBESAT_HEIGHT/2],                                        # SP Near Edge - Front
        [CUBESAT_WIDTH/2 + SOLARPANEL_WIDTH, 0, -CUBESAT_HEIGHT/2], [CUBESAT_WIDTH/2 + SOLARPANEL_WIDTH, 0, CUBESAT_HEIGHT/2],  # SP Far Edge - Front
        [CUBESAT_WIDTH/2, 0, -CUBESAT_HEIGHT/2], [CUBESAT_WIDTH/2, 0, CUBESAT_HEIGHT/2],                                        # SP Near Edge - Back
        [-CUBESAT_WIDTH/2 - SOLARPANEL_WIDTH, 0, -CUBESAT_HEIGHT/2], [-CUBESAT_WIDTH/2 - SOLARPANEL_WIDTH, 0, CUBESAT_HEIGHT/2],  # SP Far Edge - Back
    ])
    
    
def project_prism(vertices, psi, theta, phi):
    R = create_rotation_matrix(psi, theta, phi)
    rotated_vertices = np.dot(vertices, R.T)
    projected_vertices = rotated_vertices[:, 1:]  # y and z axes
    return rotated_vertices, projected_vertices



def calculate_projected_area(projected_vertices):
    # Use ConvexHull to get the vertices that form the outer polygon
    hull = ConvexHull(projected_vertices)
    
    # Extract the vertices that form the convex hull
    hull_vertices = projected_vertices[hull.vertices]

    # Calculate the area using the shoelace formula
    n = len(hull_vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull_vertices[i][0] * hull_vertices[j][1]
        area -= hull_vertices[j][0] * hull_vertices[i][1]
    
    return abs(area) / 2.0


def plot_rotation_and_projection_cube(original_vertices, rotated_vertices, projected_vertices, title):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Function to create faces for any shape
    def create_faces(vertices):
        hull = ConvexHull(vertices)
        return [vertices[simplex] for simplex in hull.simplices]
    
    # Plot original shape
    original_faces = create_faces(original_vertices)
    ax.add_collection3d(Poly3DCollection(original_faces, alpha=0.2, facecolor='green'))
    
    # Plot rotated shape
    rotated_faces = create_faces(rotated_vertices)
    ax.add_collection3d(Poly3DCollection(rotated_faces, alpha=0.2, facecolor='blue'))
    
    # Plot projected area
    hull_2d = ConvexHull(projected_vertices)
    projected_2d = projected_vertices[hull_2d.vertices]
    projected_3d = np.column_stack((np.zeros(len(projected_2d)), projected_2d))
    projected_poly = Poly3DCollection([projected_3d], alpha=0.3)
    projected_poly.set_facecolor('red')
    ax.add_collection3d(projected_poly)
    
    # Plot vertices
    ax.scatter(original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2], 
               c='g', marker='^', s=50, label='Original Vertices')
    ax.scatter(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], 
               c='b', marker='o', s=50, label='Rotated Vertices')
    ax.scatter(np.zeros_like(projected_vertices[:, 0]), projected_vertices[:, 0], projected_vertices[:, 1], 
               c='r', marker='s', s=50, label='Projected Vertices')
    
    # Draw lines from rotated vertices to their projections
    for rv, pv in zip(rotated_vertices, projected_vertices):
        ax.plot([rv[0], 0], [rv[1], pv[0]], [rv[2], pv[1]], 'k--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add a legend
    ax.legend()
    
    # Set aspect ratio to be equal
    all_vertices = np.vstack((original_vertices, rotated_vertices))
    max_range = np.array([all_vertices[:, 0].max()-all_vertices[:, 0].min(),
                          all_vertices[:, 1].max()-all_vertices[:, 1].min(),
                          all_vertices[:, 2].max()-all_vertices[:, 2].min()]).max() / 2.0
    mid_x = (all_vertices[:, 0].max()+all_vertices[:, 0].min()) * 0.5
    mid_y = (all_vertices[:, 1].max()+all_vertices[:, 1].min()) * 0.5
    mid_z = (all_vertices[:, 2].max()+all_vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add the YZ plane (projection plane)
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    yz_plane = plt.Rectangle((ylim[0], zlim[0]), ylim[1]-ylim[0], zlim[1]-zlim[0], 
                             color='c', alpha=0.1, transform=ax.get_yaxis_transform())
    ax.add_patch(yz_plane)
    art3d.pathpatch_2d_to_3d(yz_plane, z=0, zdir='x')
    
    plt.tight_layout()
    plt.show()

def plot_rotation_and_projection(original_vertices, rotated_vertices, projected_vertices, title):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the faces of the satellite
    cube_faces = [
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [0, 2, 6, 4],  # Left face
        [1, 3, 7, 5],  # Right face
        [0, 1, 3, 2],  # Bottom face
        [4, 5, 7, 6],  # Top face
    ]
    
    solar_panel_faces = [
        [8, 9, 11, 10],  # Front solar panel
        [12, 13, 15, 14],  # Back solar panel
    ]
    
    # Plot original shape
    for face in cube_faces:
        ax.add_collection3d(Poly3DCollection([original_vertices[face]], alpha=0.2, facecolor='green'))
    for face in solar_panel_faces:
        ax.add_collection3d(Poly3DCollection([original_vertices[face]], alpha=0.2, facecolor='cyan'))
    
    # Plot rotated shape
    for face in cube_faces:
        ax.add_collection3d(Poly3DCollection([rotated_vertices[face]], alpha=0.2, facecolor='blue'))
    for face in solar_panel_faces:
        ax.add_collection3d(Poly3DCollection([rotated_vertices[face]], alpha=0.2, facecolor='magenta'))
    
    # Plot projected area
    hull_2d = ConvexHull(projected_vertices)
    projected_2d = projected_vertices[hull_2d.vertices]
    projected_3d = np.column_stack((np.zeros(len(projected_2d)), projected_2d))
    projected_poly = Poly3DCollection([projected_3d], alpha=0.3)
    projected_poly.set_facecolor('red')
    ax.add_collection3d(projected_poly)
    
    # Plot vertices
    ax.scatter(original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2], 
               c='g', marker='^', s=50, label='Original Vertices')
    ax.scatter(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], 
               c='b', marker='o', s=50, label='Rotated Vertices')
    ax.scatter(np.zeros_like(projected_vertices[:, 0]), projected_vertices[:, 0], projected_vertices[:, 1], 
               c='r', marker='s', s=50, label='Projected Vertices')
    
    # Draw lines from rotated vertices to their projections
    for rv, pv in zip(rotated_vertices, projected_vertices):
        ax.plot([rv[0], 0], [rv[1], pv[0]], [rv[2], pv[1]], 'k--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add a legend
    ax.legend()
    
    # Set aspect ratio to be equal
    all_vertices = np.vstack((original_vertices, rotated_vertices))
    max_range = np.array([all_vertices[:, 0].max()-all_vertices[:, 0].min(),
                          all_vertices[:, 1].max()-all_vertices[:, 1].min(),
                          all_vertices[:, 2].max()-all_vertices[:, 2].min()]).max() / 2.0
    mid_x = (all_vertices[:, 0].max()+all_vertices[:, 0].min()) * 0.5
    mid_y = (all_vertices[:, 1].max()+all_vertices[:, 1].min()) * 0.5
    mid_z = (all_vertices[:, 2].max()+all_vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add the YZ plane (projection plane)
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    yz_plane = plt.Rectangle((ylim[0], zlim[0]), ylim[1]-ylim[0], zlim[1]-zlim[0], 
                             color='c', alpha=0.1, transform=ax.get_yaxis_transform())
    ax.add_patch(yz_plane)
    art3d.pathpatch_2d_to_3d(yz_plane, z=0, zdir='x')
    
    plt.tight_layout()
    plt.show()


def test_projected_area_simple_cube():
    test_cases = [
        {
            "name": "Unrotated shape",
            "psi": 0, "theta": 0, "phi": 0,
            "expected_area": 1.0  # Adjust this based on your initial shape
        },
        {
            "name": "45-degree rotation around z-axis",
            "psi": np.pi/4, "theta": 0, "phi": 0,
            "expected_area": 1.0  # Adjust this based on your shape
        },
        {
            "name": "90-degree rotation around y-axis",
            "psi": 0, "theta": np.pi/2, "phi": 0,
            "expected_area": 1.0  # Adjust this based on your shape
        },
        {
            "name": "45-degree rotation around x-axis",
            "psi": 0, "theta": 0, "phi": np.pi/4,
            "expected_area": 1.0  # Adjust this based on your shape
        },
        {
            "name": "Complex rotation",
            "psi": np.pi/6, "theta": np.pi/4, "phi": np.pi/3,
            "expected_area": 1.0  # Adjust this based on your shape
        }
    ]

    params = Parameters()  # Create a Parameters instance

    for case in test_cases:
        # Get the original vertices of your shape
        original_vertices = define_satellite_vertices_simple(params)

        # Rotate and project the vertices
        rotated_vertices, projected_vertices = project_prism(original_vertices, case["psi"], case["theta"], case["phi"])
        
        print(f"\nTest Case: {case['name']}")
        print(f"Rotation angles: psi={case['psi']:.2f}, theta={case['theta']:.2f}, phi={case['phi']:.2f}")
        print(f"Original vertices:\n{original_vertices}")
        print(f"Rotated vertices:\n{rotated_vertices}")
        print(f"Projected vertices:\n{projected_vertices}")
        
        calculated_area = calculate_projected_area(projected_vertices)
        
        print(f"Expected area: {case['expected_area']:.6f}")
        print(f"Calculated area: {calculated_area:.6f}")
        
        # Check if areas match
        area_match = np.isclose(case['expected_area'], calculated_area, rtol=1e-5)
        
        print(f"Area match: {area_match}")
        
        # Create 3D plot for this test case
        plot_rotation_and_projection_cube(original_vertices, rotated_vertices, projected_vertices, f"Rotation: {case['name']}")

def test_projected_area():
    test_cases = [
        {
            "name": "Unrotated shape",
            "psi": 0, "theta": 0, "phi": 0,
        },
        {
            "name": "45-degree rotation around z-axis",
            "psi": np.pi/4, "theta": 0, "phi": 0,
        },
        {
            "name": "90-degree rotation around y-axis",
            "psi": 0, "theta": np.pi/2, "phi": 0,
        },
        {
            "name": "45-degree rotation around x-axis",
            "psi": 0, "theta": 0, "phi": np.pi/4,
        },
        {
            "name": "Complex rotation",
            "psi": np.pi/6, "theta": np.pi/4, "phi": np.pi/3,
        }
    ]

    params = Parameters()  # Create a Parameters instance

    # Calculate the reference area (projected area of unrotated shape)
    original_vertices = define_satellite_vertices(params)
    _, reference_projection = project_prism(original_vertices, 0, 0, 0)
    reference_area = calculate_projected_area(reference_projection)

    for case in test_cases:
        # Get the original vertices of your shape
        original_vertices = define_satellite_vertices(params)

        # Rotate and project the vertices
        rotated_vertices, projected_vertices = project_prism(original_vertices, case["psi"], case["theta"], case["phi"])
        
        print(f"\nTest Case: {case['name']}")
        print(f"Rotation angles: psi={case['psi']:.2f}, theta={case['theta']:.2f}, phi={case['phi']:.2f}")
        
        calculated_area = calculate_projected_area(projected_vertices)
        
        print(f"Reference area: {reference_area:.6f}")
        print(f"Calculated area: {calculated_area:.6f}")
        print(f"Area ratio (calculated/reference): {calculated_area/reference_area:.6f}")
        
        # Create 3D plot for this test case
        plot_rotation_and_projection(original_vertices, rotated_vertices, projected_vertices, f"Rotation: {case['name']}")




if __name__ == "__main__":
    # Run test cases
    
    # Uncomment below line to do tests on simple cube rotation:
    #test_projected_area_simple_cube() 
    
    # Run test cases for complex satellite shape
    test_projected_area()