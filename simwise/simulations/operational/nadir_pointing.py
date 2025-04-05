import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.utils.plots import plot_results, plot_states_plotly
from simwise.simulations.base import run_dispersions, run_one
from simwise.math.quaternion import quaternion_to_dcm

def visualize_attitude_dynamics(states):
    """
    Create an animated 3D visualization of the satellite's attitude dynamics.
    
    Args:
        states: Array of satellite states containing quaternions and angular velocities
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set plot limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    # Initialize coordinate system arrows
    x_arrow, = ax.plot([], [], [], 'r-', lw=2, label='X-axis')
    y_arrow, = ax.plot([], [], [], 'g-', lw=2, label='Y-axis')
    z_arrow, = ax.plot([], [], [], 'b-', lw=2, label='Z-axis')
    
    # Add legend
    ax.legend()
    
    def update(frame):
        # Get quaternion for current frame (multiply by 10 to skip frames)
        frame_idx = frame * 10
        if frame_idx >= len(states):  # Prevent index out of bounds
            frame_idx = len(states) - 1
            
        q = states[frame_idx].q
        
        # Convert quaternion to DCM
        dcm = quaternion_to_dcm(q)
        
        # Create unit vectors
        x_unit = np.array([1, 0, 0])
        y_unit = np.array([0, 1, 0])
        z_unit = np.array([0, 0, 1])
        
        # Transform unit vectors using DCM
        x_rotated = dcm @ x_unit
        y_rotated = dcm @ y_unit
        z_rotated = dcm @ z_unit
        
        # Update arrow positions
        x_arrow.set_data([0, x_rotated[0]], [0, x_rotated[1]])
        x_arrow.set_3d_properties([0, x_rotated[2]])
        
        y_arrow.set_data([0, y_rotated[0]], [0, y_rotated[1]])
        y_arrow.set_3d_properties([0, y_rotated[2]])
        
        z_arrow.set_data([0, z_rotated[0]], [0, z_rotated[1]])
        z_arrow.set_3d_properties([0, z_rotated[2]])
        
        # Update title with time
        ax.set_title(f'Time: {states[frame_idx].t:.1f} seconds')
        
        # Return all artists that need to be redrawn
        return x_arrow, y_arrow, z_arrow, ax
    
    # Create animation with reduced number of frames
    num_frames = len(states) // 10  # Divide by 10 since we're skipping frames
    anim = FuncAnimation(
        fig, 
        update, 
        frames=num_frames,
        interval=1,  # 1ms between frames
        blit=False  # Set to False to allow title updates
    )
    
    plt.show()
    
    return anim

def run():
    # Example Usage
    overrides = {
        # for quaternions, set mean to 0 (because we are rotating the quaternion by random Euler angles)
        "q_initial": QuaternionParameter(np.array([1, 0, 0, 0]), variance=np.array([0.1, 0.1, 0.1])),
        "pointing_mode": "NadirPointing",

        "Cp": ArrayParameter(np.array([0.1, 0.2, 0.3]), mean=np.array([0.1, 0.2, 0.3]), variance=0.003),
        "Cg": ArrayParameter(np.array([0.2, 0.3, 0.4]), mean=np.array([0.2, 0.3, 0.4]), variance=0.003),
        "num_dispersions": 16,
        "dt_orbit": 120,
        "t_end": 90 * 60,
    }
    params = Parameters(**overrides)
    states, times = run_dispersions(params, runner=run_one)

    # Add visualization for the first dispersion case
    visualize_attitude_dynamics(states[0])

    plot_results(states)
    
    # Plot sun vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "x": lambda state: state.r_sun_eci[0] / np.linalg.norm(state.r_sun_eci),
            "y": lambda state: state.r_sun_eci[1] / np.linalg.norm(state.r_sun_eci),
            "z": lambda state: state.r_sun_eci[2] / np.linalg.norm(state.r_sun_eci),
        },
        spacing=0.05,
        title_text="Sun Vector vs Time",
    )
    fig_mag.show()

    # Plot e_angles
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "x": lambda state: state.e_angles[0] / np.linalg.norm(state.e_angles),
            "y": lambda state: state.e_angles[1] / np.linalg.norm(state.e_angles),
            "z": lambda state: state.e_angles[2] / np.linalg.norm(state.e_angles),
        },
        spacing=0.05,
        title_text="Euler Angles vs Time",
    )
    fig_mag.show()

    # Plot q_d
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "q_d0": lambda state: state.q_d[0] / np.linalg.norm(state.q_d),
            "q_d1": lambda state: state.q_d[1] / np.linalg.norm(state.q_d),
            "q_d2": lambda state: state.q_d[2] / np.linalg.norm(state.q_d),
            "q_d3": lambda state: state.q_d[3] / np.linalg.norm(state.q_d),
        },
        spacing=0.05,
        title_text="q vs q_d vs time",
    )
    fig_mag.show()

    # Plot w_d
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "w_d0": lambda state: state.w_d[0] / np.linalg.norm(state.w_d),
            "w_d1": lambda state: state.w_d[1] / np.linalg.norm(state.w_d),
            "w_d2": lambda state: state.w_d[2] / np.linalg.norm(state.w_d),
            "w_d3": lambda state: state.w_d[3] / np.linalg.norm(state.w_d),
        },
        spacing=0.05,
        title_text="w vs w_d vs time",
    )
    fig_mag.show()

    # Plot Satellite Coordinate Vectors
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "x": lambda state: state.r_eci[0] / np.linalg.norm(state.r_eci),
            "y": lambda state: state.r_eci[1] / np.linalg.norm(state.r_eci),
            "z": lambda state: state.r_eci[2] / np.linalg.norm(state.r_eci),
        },
        spacing=0.05,
        title_text="Nadir Vector vs Time",
    )
    fig_mag.show()
    
