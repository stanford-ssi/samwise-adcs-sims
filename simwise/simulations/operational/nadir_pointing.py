import numpy as np
from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.utils.plots import plot_results, plot_states_plotly
from simwise.simulations.base import run_dispersions, run_one
from simwise.visualization.attitude_visualizer import visualize_attitude_dynamics
from simwise.visualization.orbit_attitude_visualizer import visualize_orbit_and_attitude
import matplotlib.pyplot as plt

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
    }
    params = Parameters(**overrides)
    states, times = run_dispersions(params, runner=run_one)

    # Show both visualizations
    visualize_attitude_dynamics(states[0])
    visualize_orbit_and_attitude(states[0])

    plot_results(states)
    
    # Plot sun vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state .jd,
        {
            "x": lambda state: np.array(state.r_sun_eci)[0] / np.linalg.norm(state.r_sun_eci),
            "y": lambda state: np.array(state.r_sun_eci)[1] / np.linalg.norm(state.r_sun_eci),
            "z": lambda state: np.array(state.r_sun_eci)[2] / np.linalg.norm(state.r_sun_eci),
        },
        spacing=0.05,
        title_text="Sun Vector vs Time",
    )
    fig_mag.show()

    # Plot e_angles (convert to degrees)
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "θ": lambda state: np.degrees(state.e_angles[0]) / np.linalg.norm(np.degrees(state.e_angles)),
            "ϕ": lambda state: np.degrees(state.e_angles[1]) / np.linalg.norm(np.degrees(state.e_angles)),
            "ψ": lambda state: np.degrees(state.e_angles[2]) / np.linalg.norm(np.degrees(state.e_angles)),
        },
        spacing=0.05,
        title_text="Euler Angles vs Time (degrees)",
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


    # Plot Satellite Coordinate Vectors
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "x": lambda state: np.array(state.r_eci)[0] / np.linalg.norm(state.r_eci),
            "y": lambda state: np.array(state.r_eci)[1] / np.linalg.norm(state.r_eci),
            "z": lambda state: np.array(state.r_eci)[2] / np.linalg.norm(state.r_eci),
        },
        spacing=0.05,
        title_text="Satellite ECI Position vs Time [s]",
    )
    fig_mag.show()
    
    # Extract nadir vectors (negative normalized r_eci)
    nadir_vectors = np.array([
        (-np.array(state.r_eci) / np.linalg.norm(state.r_eci)) 
        for state in states[0]
    ])
    
    # DEBUG: Add diagnostic plots for target attitude implementation
    print("\nDEBUG: Analyzing target attitude computation")
    
    # Create arrays for time and relevant quantities
    times = [state.t for state in states[0]]
    q_d_norms = [np.linalg.norm(state.q_d) for state in states[0]]
    error_angles = [state.error_angle if hasattr(state, 'error_angle') else 0 for state in states[0]]
    
    # Plot quaternion norm over time (should always be ~1)
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(times, q_d_norms, 'b-')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.title('Target Quaternion Norm vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('||q_d||')
    plt.grid(True)

    # Plot error angle over time
    plt.subplot(132)
    plt.plot(times, error_angles, 'b-')
    plt.title('Attitude Error vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Error Angle [rad]')
    plt.grid(True)

    # Plot individual q_d components
    plt.subplot(133)
    q_d_components = np.array([state.q_d for state in states[0]])
    for i in range(4):
        plt.plot(times, q_d_components[:, i], label=f'q_d[{i}]')
    plt.title('Target Quaternion Components')
    plt.xlabel('Time [s]')
    plt.ylabel('Component Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Create a dedicated plot for the nadir vector from the function and the target attitude
    plt.figure(figsize=(12, 10))
    
    # Extract nadir vectors (both the one we calculated and the one from the function)
    # Note: We may need to check if the attribute exists
    computed_nadir_vectors = np.array([
        getattr(state, 'nadir_eci_computed', np.zeros(3)) 
        for state in states[0]
    ])
    
    # Plot both nadir vectors for comparison (if computed_nadir_vectors exists)
    if np.any(computed_nadir_vectors):
        plt.subplot(411)
        plt.plot(times, nadir_vectors[:, 0], 'r-', label='Calculated Nadir X')
        plt.plot(times, nadir_vectors[:, 1], 'g-', label='Calculated Nadir Y')
        plt.plot(times, nadir_vectors[:, 2], 'b-', label='Calculated Nadir Z')
        plt.plot(times, computed_nadir_vectors[:, 0], 'r--', label='Function Nadir X')
        plt.plot(times, computed_nadir_vectors[:, 1], 'g--', label='Function Nadir Y')
        plt.plot(times, computed_nadir_vectors[:, 2], 'b--', label='Function Nadir Z')
        plt.title('Nadir Vector Components Comparison')
        plt.xlabel('Time [s]')
        plt.ylabel('Component Value')
        plt.legend()
        plt.grid(True)
    else:
        # Just plot the calculated nadir vector 
        plt.subplot(411)
        plt.plot(times, nadir_vectors[:, 0], 'r-', label='Nadir X')
        plt.plot(times, nadir_vectors[:, 1], 'g-', label='Nadir Y')
        plt.plot(times, nadir_vectors[:, 2], 'b-', label='Nadir Z')
        plt.title('Nadir Vector Components vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Component Value')
        plt.legend()
        plt.grid(True)
    
    # Plot nadir vector magnitude (should always be 1)
    plt.subplot(412)
    nadir_magnitudes = np.linalg.norm(nadir_vectors, axis=1)
    plt.plot(times, nadir_magnitudes, 'k-')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.title('Nadir Vector Magnitude vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('||Nadir||')
    plt.ylim(0.9, 1.1)  # Tight around expected value of 1
    plt.grid(True)
    
    # Plot target attitude quaternion components
    plt.subplot(413)
    q_d_components = np.array([state.q_d for state in states[0]])
    for i in range(4):
        plt.plot(times, q_d_components[:, i], label=f'q_d[{i}]')
    plt.title('Target Attitude Quaternion Components')
    plt.xlabel('Time [s]')
    plt.ylabel('Component Value')
    plt.legend()
    plt.grid(True)
    
    # Plot difference between consecutive quaternion components to check for jumps
    plt.subplot(414)
    q_d_diff = np.diff(q_d_components, axis=0)
    for i in range(4):
        plt.plot(times[1:], q_d_diff[:, i], label=f'Δq_d[{i}]')
    plt.title('Changes in Target Quaternion Components')
    plt.xlabel('Time [s]')
    plt.ylabel('Component Difference')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Add this additional debug plot for Euler angles in degrees
    plt.figure(figsize=(10, 6))
    plt.subplot(111)
    
    # Extract Euler angles and convert to degrees
    euler_angles_deg = np.array([np.degrees(state.e_angles) for state in states[0]])
    
    # Plot each Euler angle component
    plt.plot(times, euler_angles_deg[:, 0], 'r-', label='Roll (ϕ)')
    plt.plot(times, euler_angles_deg[:, 1], 'g-', label='Pitch (θ)')
    plt.plot(times, euler_angles_deg[:, 2], 'b-', label='Yaw (ψ)')
    
    plt.title('Euler Angles vs Time (degrees)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [degrees]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
