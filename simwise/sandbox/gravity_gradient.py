import numpy as np
from simwise.constants import *
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# GRAVITY GRADIENT CALCULATION:

def gravityGradientPerturbationTorque(Cp, Cg, e_angles, r_vector, inertia):
    """
    Calculate the gravity gradient perturbation torque on the satellite.

    Args:
        Cp (np.ndarray): Center of pressure, shape (3,)
        Cg (np.ndarray): Center of gravity, shape (3,)
        e_angles (np.ndarray): Euler angles [psi, theta, phi], shape (3,)
        r_vector (np.ndarray): Position vector in ECI frame, shape (3,)
        inertia (np.ndarray): Principal moments of inertia, shape (3,)

    Returns:
        np.ndarray: Gravity gradient perturbation torque, shape (3,)
    """
    # Earth's gravitational parameter
    mu_earth = 3.986004418e14  # [m^3/s^2]

    # Calculate rotation matrix from Euler angles
    R = create_rotation_matrix(e_angles[0], e_angles[1], e_angles[2])

    # Transform position vector to body frame
    r_body = np.dot(R, r_vector)

    # Calculate unit vector in direction of position vector
    r_mag = np.linalg.norm(r_body)
    r_hat = r_body / r_mag

    # Calculate differences in moments of inertia
    I_diff = np.array([
        inertia[1] - inertia[2],
        inertia[2] - inertia[0],
        inertia[0] - inertia[1]
    ])

    # Calculate gravity gradient coefficient
    coeff = 3 * mu_earth / (2 * r_mag**3)

    # Calculate r_hat cross terms
    r_cross = np.array([
        r_hat[1] * r_hat[2],
        r_hat[2] * r_hat[0],
        r_hat[0] * r_hat[1]
    ])

    # Calculate gravity gradient torque
    torque = coeff * I_diff * r_cross

    return torque

def create_rotation_matrix(psi, theta, phi):
    """
    Create rotation matrix from Euler angles (ZYX sequence).

    Args:
        psi (float): Rotation around Z axis [rad]
        theta (float): Rotation around Y axis [rad]
        phi (float): Rotation around X axis [rad]

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
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

def plot_gravity_gradient_visualization(r_vector, inertia, e_angles, torque):
    """
    Create a 3D visualization of the gravity gradient torque.

    Args:
        r_vector (np.ndarray): Position vector in ECI frame
        inertia (np.ndarray): Principal moments of inertia
        e_angles (np.ndarray): Euler angles [psi, theta, phi]
        torque (np.ndarray): Calculated gravity gradient torque
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot coordinate system
    origin = np.zeros(3)
    ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='r', label='X')
    ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='g', label='Y')
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='b', label='Z')

    # Plot position vector
    r_mag = np.linalg.norm(r_vector)
    ax.quiver(origin[0], origin[1], origin[2],
             r_vector[0], r_vector[1], r_vector[2],
             color='k', label='Position Vector')

    # Plot torque vector (scaled for visualization)
    torque_scale = r_mag / np.linalg.norm(torque) * 0.2
    ax.quiver(origin[0], origin[1], origin[2],
             torque[0] * torque_scale,
             torque[1] * torque_scale,
             torque[2] * torque_scale,
             color='m', label='Gravity Gradient Torque')

    # Plot inertia ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.sqrt(inertia[0]) * np.outer(np.cos(u), np.sin(v))
    y = np.sqrt(inertia[1]) * np.outer(np.sin(u), np.sin(v))
    z = np.sqrt(inertia[2]) * np.outer(np.ones_like(u), np.cos(v))

    # Rotate the ellipsoid according to euler angles
    R = create_rotation_matrix(e_angles[0], e_angles[1], e_angles[2])
    for i in range(len(x)):
        for j in range(len(x)):
            point = np.dot(R, np.array([x[i,j], y[i,j], z[i,j]]))
            x[i,j], y[i,j], z[i,j] = point

    ax.plot_surface(x, y, z, alpha=0.2, color='c')

    # Set labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Gravity Gradient Torque Visualization')

    # Add legend
    ax.legend()

    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

def test_gravity_gradient():
    """
    Test the gravity gradient torque calculation with various scenarios.
    """
    test_cases = [
        {
            "name": "Nominal case",
            "e_angles": np.array([0, 0, 0]),
            "r_vector": np.array([7000e3, 0, 0]),  # Satellite on x-axis
        },
        {
            "name": "45-degree rotation",
            "e_angles": np.array([np.pi/4, np.pi/4, np.pi/4]),
            "r_vector": np.array([7000e3/np.sqrt(2), 7000e3/np.sqrt(2), 0]),
        },
        {
            "name": "Complex orientation",
            "e_angles": np.array([np.pi/6, np.pi/3, np.pi/4]),
            "r_vector": np.array([7000e3/np.sqrt(3), 7000e3/np.sqrt(3), 7000e3/np.sqrt(3)]),
        }
    ]

    # Test parameters
    Cp = np.array([0, 0, 0])
    Cg = np.array([20/4, 10*np.sqrt(2)/2, 10*np.sqrt(2)/2])
    inertia = np.array([0.01461922201, 0.0412768466, 0.03235309961])

    for case in test_cases:
        print(f"\nTest Case: {case['name']}")
        print(f"Euler angles: {case['e_angles']}")
        print(f"Position vector: {case['r_vector']}")

        # Calculate torque
        torque = gravityGradientPerturbationTorque(
            Cp, Cg, case['e_angles'], case['r_vector'], inertia
        )

        print(f"Calculated torque: {torque}")

        # Create visualization
        plot_gravity_gradient_visualization(
            case['r_vector'], inertia, case['e_angles'], torque
        )

if __name__ == "__main__":
    # Run test cases
    test_gravity_gradient()