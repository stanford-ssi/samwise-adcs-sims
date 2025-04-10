import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from simwise.math.quaternion import quaternion_to_dcm

def visualize_orbit_and_attitude(states):
    """
    Create an animated 3D visualization of the satellite's orbit and attitude.
    Shows Earth, orbital path, and satellite body frame.
    
    Args:
        states: Array of satellite states containing position and attitude
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # Draw Earth
    r_earth = 6.371e6  # Earth radius in meters
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    earth = ax.plot_surface(x, y, z, color='b', alpha=0.1)
    
    # Create orbit path from all positions
    orbit_x = [state.r_eci[0] for state in states]
    orbit_y = [state.r_eci[1] for state in states]
    orbit_z = [state.r_eci[2] for state in states]
    ax.plot(orbit_x, orbit_y, orbit_z, 'g--', alpha=0.5, label='Orbit')
    
    # Initialize satellite position, attitude arrows, and nadir vector
    sat_pos, = ax.plot([], [], [], 'ko', label='Satellite')
    x_arrow, = ax.plot([], [], [], 'r-', lw=2, label='X-axis')
    y_arrow, = ax.plot([], [], [], 'g-', lw=2, label='Y-axis')
    z_arrow, = ax.plot([], [], [], 'b-', lw=2, label='Z-axis')
    nadir_line, = ax.plot([], [], [], 'k--', lw=1, label='Nadir Vector')  # Add nadir vector line
    
    # Set plot limits based on orbit size
    max_range = max(max(np.abs(orbit_x)), max(np.abs(orbit_y)), max(np.abs(orbit_z)))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range]) 
    ax.set_zlim([-max_range, max_range])
    
    # Add legend with fixed location instead of "best"
    ax.legend(loc='upper right')
    
    def update(frame):
        # Get position and quaternion for current frame
        frame_idx = frame * 100
        if frame_idx >= len(states):
            frame_idx = len(states) - 1
            
        state = states[frame_idx]
        pos = state.r_eci
        q = state.q
        
        # Convert quaternion to DCM
        dcm = quaternion_to_dcm(q)
        
        # Create unit vectors (scaled for visibility)
        scale = 5e5  # Adjust this to make arrows visible relative to orbit size
        x_unit = np.array([1, 0, 0]) * scale
        y_unit = np.array([0, 1, 0]) * scale
        z_unit = np.array([0, 0, 1]) * scale
        
        # Transform unit vectors using DCM
        x_rotated = dcm @ x_unit
        y_rotated = dcm @ y_unit
        z_rotated = dcm @ z_unit
        
        # Update satellite position
        sat_pos.set_data([pos[0]], [pos[1]])
        sat_pos.set_3d_properties([pos[2]])
        
        # Update arrow positions (offset by satellite position)
        x_arrow.set_data([pos[0], pos[0] + x_rotated[0]], [pos[1], pos[1] + x_rotated[1]])
        x_arrow.set_3d_properties([pos[2], pos[2] + x_rotated[2]])
        
        y_arrow.set_data([pos[0], pos[0] + y_rotated[0]], [pos[1], pos[1] + y_rotated[1]])
        y_arrow.set_3d_properties([pos[2], pos[2] + y_rotated[2]])
        
        z_arrow.set_data([pos[0], pos[0] + z_rotated[0]], [pos[1], pos[1] + z_rotated[1]])
        z_arrow.set_3d_properties([pos[2], pos[2] + z_rotated[2]])
        
        # Update nadir vector line (from satellite to Earth center)
        nadir_line.set_data([0, pos[0]], [0, pos[1]])  # Line from origin to satellite
        nadir_line.set_3d_properties([0, pos[2]])
        
        # Update title with time
        ax.set_title(f'Time: {state.t:.1f} seconds')
        
        return sat_pos, x_arrow, y_arrow, z_arrow, nadir_line
    
    # Create animation
    num_frames = len(states) // 100
    anim = FuncAnimation(
        fig, 
        update, 
        frames=num_frames,
        interval=1,
        blit=False
    )
    
    plt.show()
    return anim