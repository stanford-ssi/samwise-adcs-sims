import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from simwise.math.quaternion import quaternion_to_dcm, quaternion_to_euler
from matplotlib.gridspec import GridSpec

def visualize_attitude_dynamics(states):
    """
    Create an animated 3D visualization of the satellite's attitude dynamics.
    Shows body frame axes and Euler angles relative to reference axes.
    Includes real-time plots of Euler angles and quaternion components.
    
    Args:
        states: Array of satellite states containing quaternions and angular velocities
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    
    # 3D visualization area
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Euler angles plot
    ax_euler = fig.add_subplot(gs[0, 1])
    ax_euler.set_title('Euler Angles vs Time (degrees)')
    ax_euler.set_xlabel('Time [s]')
    ax_euler.set_ylabel('Angle [°]')
    ax_euler.grid(True)
    
    # Quaternion components plot
    ax_quat = fig.add_subplot(gs[1, 1])
    ax_quat.set_title('Quaternion Components vs Time')
    ax_quat.set_xlabel('Time [s]')
    ax_quat.set_ylabel('Component Value')
    ax_quat.grid(True)
    
    # Prepare 3D visualization
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_xlim([-1.5, 1.5])
    ax_3d.set_ylim([-1.5, 1.5])
    ax_3d.set_zlim([-1.5, 1.5])
    
    # Initialize coordinate system arrows for body frame
    x_arrow, = ax_3d.plot([], [], [], 'r-', lw=2, label='Body X-axis')
    y_arrow, = ax_3d.plot([], [], [], 'g-', lw=2, label='Body Y-axis')
    z_arrow, = ax_3d.plot([], [], [], 'b-', lw=2, label='Body Z-axis')
    
    # Initialize reference frame (dashed translucent lines)
    x_ref, = ax_3d.plot([0, 1], [0, 0], [0, 0], 'r--', lw=1, alpha=0.3, label='Ref X-axis')
    y_ref, = ax_3d.plot([0, 0], [0, 1], [0, 0], 'g--', lw=1, alpha=0.3, label='Ref Y-axis')
    z_ref, = ax_3d.plot([0, 0], [0, 0], [0, 1], 'b--', lw=1, alpha=0.3, label='Ref Z-axis')
    
    # Initialize Euler angle arcs
    phi_arc, = ax_3d.plot([], [], [], 'r-', lw=1, label='ϕ (Roll)')
    theta_arc, = ax_3d.plot([], [], [], 'g-', lw=1, label='θ (Pitch)')
    psi_arc, = ax_3d.plot([], [], [], 'b-', lw=1, label='ψ (Yaw)')
    
    # Initialize text for displaying Euler angle values
    euler_text = ax_3d.text2D(0.05, 0.95, "", transform=ax_3d.transAxes)
    
    # Add legend to 3D plot
    ax_3d.legend(loc='upper right', fontsize='small')
    
    # Initialize time arrays for plots
    buffer_size = 100  # Number of points to show in the plots
    time_buffer = np.zeros(buffer_size)
    
    # Initialize Euler angle buffers
    roll_buffer = np.zeros(buffer_size)
    pitch_buffer = np.zeros(buffer_size)
    yaw_buffer = np.zeros(buffer_size)
    
    # Initialize quaternion component buffers
    q0_buffer = np.zeros(buffer_size)
    q1_buffer = np.zeros(buffer_size)
    q2_buffer = np.zeros(buffer_size)
    q3_buffer = np.zeros(buffer_size)
    
    # Initialize plot lines for Euler angles
    roll_line, = ax_euler.plot([], [], 'r-', label='Roll (ϕ)')
    pitch_line, = ax_euler.plot([], [], 'g-', label='Pitch (θ)')
    yaw_line, = ax_euler.plot([], [], 'b-', label='Yaw (ψ)')
    ax_euler.legend()
    
    # Initialize plot lines for quaternion components
    q0_line, = ax_quat.plot([], [], 'k-', label='q[0]')
    q1_line, = ax_quat.plot([], [], 'r-', label='q[1]')
    q2_line, = ax_quat.plot([], [], 'g-', label='q[2]')
    q3_line, = ax_quat.plot([], [], 'b-', label='q[3]')
    ax_quat.legend()
    
    # Set initial y-axis limits
    ax_euler.set_ylim([-180, 180])
    ax_quat.set_ylim([-1.1, 1.1])
    
    def create_arc_points(start_vec, end_vec, n_points=20, radius=0.5):
        """Create points for an arc between two vectors"""
        # Normalize vectors
        start_vec = start_vec / np.linalg.norm(start_vec)
        end_vec = end_vec / np.linalg.norm(end_vec)
        
        # Create orthogonal basis
        if np.allclose(start_vec, end_vec):
            return [start_vec * radius]
        
        # Calculate angle between vectors
        cos_angle = np.clip(np.dot(start_vec, end_vec), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Create points along the arc
        points = []
        for i in range(n_points):
            t = i / (n_points - 1)
            # Spherical linear interpolation
            sin_angle = np.sin(angle)
            if abs(sin_angle) < 1e-8:  # Avoid division by zero
                point = start_vec
            else:
                a = np.sin((1 - t) * angle) / sin_angle
                b = np.sin(t * angle) / sin_angle
                point = a * start_vec + b * end_vec
            points.append(point * radius)
        
        return points
    
    def update(frame):
        # Get quaternion for current frame (multiply by 10 to skip frames)
        frame_idx = frame * 10
        if frame_idx >= len(states):  # Prevent index out of bounds
            frame_idx = len(states) - 1
            
        state = states[frame_idx]
        q = state.q
        t = state.t
        
        # Convert quaternion to DCM
        dcm = quaternion_to_dcm(q)
        
        # Get Euler angles
        euler_angles = state.e_angles  # Use stored Euler angles
        
        # Create unit vectors
        x_unit = np.array([1, 0, 0])
        y_unit = np.array([0, 1, 0])
        z_unit = np.array([0, 0, 1])
        
        # Transform unit vectors using DCM
        x_rotated = dcm @ x_unit
        y_rotated = dcm @ y_unit
        z_rotated = dcm @ z_unit
        
        # Update arrow positions for body frame
        x_arrow.set_data([0, x_rotated[0]], [0, x_rotated[1]])
        x_arrow.set_3d_properties([0, x_rotated[2]])
        
        y_arrow.set_data([0, y_rotated[0]], [0, y_rotated[1]])
        y_arrow.set_3d_properties([0, y_rotated[2]])
        
        z_arrow.set_data([0, z_rotated[0]], [0, z_rotated[1]])
        z_arrow.set_3d_properties([0, z_rotated[2]])
        
        # Create arcs for Euler angles
        # Roll (phi) - rotation around X axis (shown as arc from Y to Y')
        phi_points = create_arc_points(y_unit, y_rotated, n_points=30)
        phi_x = [p[0] for p in phi_points]
        phi_y = [p[1] for p in phi_points]
        phi_z = [p[2] for p in phi_points]
        phi_arc.set_data(phi_x, phi_y)
        phi_arc.set_3d_properties(phi_z)
        
        # Pitch (theta) - rotation around Y axis (shown as arc from X to X')
        theta_points = create_arc_points(x_unit, x_rotated, n_points=30)
        theta_x = [p[0] for p in theta_points]
        theta_y = [p[1] for p in theta_points]
        theta_z = [p[2] for p in theta_points]
        theta_arc.set_data(theta_x, theta_y)
        theta_arc.set_3d_properties(theta_z)
        
        # Yaw (psi) - rotation around Z axis (shown as arc from X to X' projected on XY plane)
        psi_points = create_arc_points(
            np.array([x_unit[0], x_unit[1], 0]), 
            np.array([x_rotated[0], x_rotated[1], 0]), 
            n_points=30
        )
        psi_x = [p[0] for p in psi_points]
        psi_y = [p[1] for p in psi_points]
        psi_z = [0] * len(psi_points)  # Keep on XY plane
        psi_arc.set_data(psi_x, psi_y)
        psi_arc.set_3d_properties(psi_z)
        
        # Update Euler angle text (convert radians to degrees)
        euler_text.set_text(f"Euler Angles (ZYX):\nϕ (Roll): {np.degrees(euler_angles[0]):.1f}°\n"
                            f"θ (Pitch): {np.degrees(euler_angles[1]):.1f}°\n"
                            f"ψ (Yaw): {np.degrees(euler_angles[2]):.1f}°")
        
        # Update title with time
        ax_3d.set_title(f'Time: {t:.1f} seconds')
        
        # Update plot data
        # Shift buffers and add new data
        time_buffer[:-1] = time_buffer[1:]
        time_buffer[-1] = t
        
        # Update Euler angle buffers
        roll_buffer[:-1] = roll_buffer[1:]
        roll_buffer[-1] = np.degrees(euler_angles[0])
        
        pitch_buffer[:-1] = pitch_buffer[1:]
        pitch_buffer[-1] = np.degrees(euler_angles[1])
        
        yaw_buffer[:-1] = yaw_buffer[1:]
        yaw_buffer[-1] = np.degrees(euler_angles[2])
        
        # Update quaternion component buffers
        q0_buffer[:-1] = q0_buffer[1:]
        q0_buffer[-1] = q[0]
        
        q1_buffer[:-1] = q1_buffer[1:]
        q1_buffer[-1] = q[1]
        
        q2_buffer[:-1] = q2_buffer[1:]
        q2_buffer[-1] = q[2]
        
        q3_buffer[:-1] = q3_buffer[1:]
        q3_buffer[-1] = q[3]
        
        # Update plots
        roll_line.set_data(time_buffer, roll_buffer)
        pitch_line.set_data(time_buffer, pitch_buffer)
        yaw_line.set_data(time_buffer, yaw_buffer)
        
        q0_line.set_data(time_buffer, q0_buffer)
        q1_line.set_data(time_buffer, q1_buffer)
        q2_line.set_data(time_buffer, q2_buffer)
        q3_line.set_data(time_buffer, q3_buffer)
        
        # Adjust x-axis limits to show the current time window
        if t > 0:
            lower_t = max(0, t - 30)  # Show last 30 seconds
            ax_euler.set_xlim([lower_t, t + 5])
            ax_quat.set_xlim([lower_t, t + 5])
        
        # Return all artists that need to be redrawn
        return (x_arrow, y_arrow, z_arrow, x_ref, y_ref, z_ref, 
                phi_arc, theta_arc, psi_arc, euler_text, 
                roll_line, pitch_line, yaw_line,
                q0_line, q1_line, q2_line, q3_line)
    
    # Adjust layout
    plt.tight_layout()
    
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