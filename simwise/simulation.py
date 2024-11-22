import numpy as np
from scipy.integrate import solve_ivp
from quaternion import quaternion2euler, quaternion_dynamics, compute_control_torque, angle_axis_between
from bdot import bdot_bang_bang, bdot_step_bang_bang, bdot_proportional, bdot_pid
from plots import plot_vector_plotly, plot_quaternion_plotly
import plotly.graph_objects as go
from graph_utils import graph_euler, graph_vector_matplotlib, graph_quaternion, graph_quaternion_matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

def magnetic_field(t):
        B_magnitude = 4.5e-5  # ~45 μT, typical in LEO
        omega_orbit = 2 * np.pi / 5600  # Approximate orbital angular velocity
        
        Bx = B_magnitude * np.sin(omega_orbit * t)
        By = B_magnitude * np.cos(omega_orbit * t)
        Bz = B_magnitude * 0.5 * np.sin(2 * omega_orbit * t)
        
        return np.array([Bx, By, Bz])


def simulate_control_torques():
    # Initial conditions
    q = np.array([1, 0, 0, 0])
    Ω = np.array([0.0, 0.0, 0.1]) # [rad/s]
    x0 = np.concatenate((q, Ω))
    
    # Desired state
    q_d = np.array([0.5, 0.5, 0.5, 0.5])
    Ω_d = np.array([0, 0, 0]) # [rad/s]
    x_d = np.concatenate((q_d, Ω_d))

    # Inertia tensor [kg m^2]
    # Comnputed using moment_of_inertia.ipynb
    inertia = np.array([0.01461922201, 0.0412768466, 0.03235309961])
    
    # Control coefficients
    K_p = 0.0005
    K_d = 0.005

    # Noise parameter (standard deviation of gaussian) [Nm]
    tau_noise = 0.00000288

    # Maximum actuator torque [Nm]
    tau_max = 0.0032

    dt = 1/60 # [sec]
    t_end = 10 * 60 # [sec] 2 minutes
    epoch = 0
    num_points = int(t_end // dt)

    y = np.zeros((num_points, 7))
    t_arr = np.arange(num_points) * dt + epoch
    x = x0
    e_angles = np.zeros((num_points, 3))
    quaternions = np.zeros((num_points, 4))
    omegas = np.zeros((num_points, 3))

    print("Simulating...")
    for i in tqdm(range(num_points)):
        t = t_arr[i]

        f = lambda t, x: quaternion_dynamics(x, dt, inertia, compute_control_torque(x, x_d, K_p, K_d, tau_max=tau_max), noise=tau_noise)
        sol = solve_ivp(f, [t, t+dt], x, method='RK45')
        y[i] = sol.y[:, -1]
        x = y[i]
        e_angles[i] = quaternion2euler(y[i, :4])
        quaternions[i] = y[i, :4]
        omegas[i] = y[i, 4:]

    # Back compute values from simulation results
    theta_err = [angle_axis_between(quaternions[i], q_d)[0] for i in range(num_points)]
    axis = np.array([angle_axis_between(quaternions[i], q_d)[1] for i in range(num_points)])
    torques = np.array([compute_control_torque(y[i], x_d, K_p, K_d) for i in range(num_points)])

    # Plot results
    plt.plot(t_arr, theta_err, "r")
    plt.xlabel("Time")
    plt.ylabel("Error angle (rad)")
    plt.show()

    graph_quaternion_matplotlib(t_arr, quaternions)
    graph_vector_matplotlib(t_arr, torques, "torque x", "torque y", "torque z")
    # graph_vector_matplotlib(t_arr, axis, "error axis x", "error axis y", "error axis z")
    graph_vector_matplotlib(t_arr, omegas, "wx", "wy", "wz")

def simulate_bdot():
    # Initial conditions
    q = np.array([1, 0, 0, 0])  # Initial quaternion (no rotation)
    Ω = np.array([0.1, -0.15, 0.08])  # Initial angular velocity [rad/s]
    x0 = np.concatenate((q, Ω))
    
    # Spacecraft parameters
    # Inertia tensor [kg m^2] - using the values from your original simulation
    inertia = np.array([0.01461922201, 0.0412768466, 0.03235309961])
    
    # Maximum magnetic moment [A⋅m²]
    mu_max = 2*3.0e-2  # 0.030 A⋅m² 
    
    # Simulation parameters
    dt = 5  # [sec]
    t_end = 24 * 60 * 60  # 8 hour simulation
    num_points = int(t_end // dt)
    t_arr = np.arange(num_points) * dt
    
    # Storage arrays for plotting
    omegas = np.zeros((num_points, 3))
    magnetic_moments = np.zeros((num_points, 3))
    torques = np.zeros((num_points, 3))
    B_fields = np.zeros((num_points, 3))
    
    # Time-varying magnetic field (simulating orbital motion)
    # This creates a rotating magnetic field as might be seen in orbit (NOT ACCURATE W/ POLE BEHAVIOR)
    
    print("Simulating magnetic detumbling...")
    x = x0
    for i in tqdm(range(num_points)):
        t = t_arr[i]
        
        # Get current magnetic field
        B = magnetic_field(t)
        B_fields[i] = B
        
        # Compute control torque using B-dot control algorithm
        # tau, mu = bdot_bang_bang(x, B, mu_max)
        # tau, mu = bdot_step_bang_bang(x, B, mu_max)
        # tau, mu = bdot_proportional(x, B, mu_max)
        tau, mu = bdot_pid(x, B, mu_max)
        
        
        # Store values for plotting
        omegas[i] = x[4:]
        magnetic_moments[i] = mu
        torques[i] = tau
        
        # Propagate dynamics
        f = lambda t, x: quaternion_dynamics(x, dt, inertia, tau)
        sol = solve_ivp(f, [t, t+dt], x, method='RK45')
        x = sol.y[:, -1]
    
    # Plot results using Plotly
    # Angular velocity magnitude
    fig_mag = go.Figure()
    fig_mag.add_trace(go.Scatter(
        x=t_arr,
        y=np.linalg.norm(omegas, axis=1),
        line=dict(color='#1f77b4')
    ))
    fig_mag.update_layout(
        title="Detumbling Performance",
        xaxis_title="Time [s]",
        yaxis_title="Angular Velocity Magnitude [rad/s]"
    )
    fig_mag.show()

    # Angular velocities
    fig_omega = plot_vector_plotly(t_arr, omegas,
                                 ["ωx", "ωy", "ωz"],
                                 ylabel="Angular velocity [rad/s]")
    fig_omega.update_layout(title="Angular Velocities vs Time")
    fig_omega.show()

    # Magnetic moments
    fig_mu = plot_vector_plotly(t_arr, magnetic_moments,
                              ["μx", "μy", "μz"],
                              ylabel="Magnetic moment [A⋅m²]")
    fig_mu.update_layout(title="Magnetic Moments vs Time")
    fig_mu.show()

    # Magnetic field
    # fig_B = plot_vector_plotly(t_arr, B_fields,
    #                          ["Bx", "By", "Bz"],
    #                          ylabel="Magnetic field [T]")
    # fig_B.update_layout(title="Magnetic Field Components vs Time")
    # fig_B.show()

def evaluate_performance(omegas: np.ndarray, magnetic_moments: np.ndarray, 
                    settling_threshold: float = 0.01) -> tuple:
    """
    Evaluate the performance of a control run.
    
    Args:
        omegas: Angular velocity history [rad/s]
        magnetic_moments: Control effort history [A⋅m²]
        settling_threshold: Angular velocity magnitude considered "settled" [rad/s]
    
    Returns:
        tuple: (settling_time, control_effort, final_omega_mag)
    """
    omega_mags = np.linalg.norm(omegas, axis=1)
    
    # Find settling time (when omega magnitude stays below threshold)
    settled_indices = np.where(omega_mags < settling_threshold)[0]
    if len(settled_indices) > 0:
        # Find first index where it stays below threshold for at least 100 points
        for idx in settled_indices:
            if idx + 100 < len(omega_mags):
                if np.all(omega_mags[idx:idx+100] < settling_threshold):
                    settling_time = idx
                    break
        else:
            settling_time = len(omega_mags)  # Never truly settled
    else:
        settling_time = len(omega_mags)  # Never reached threshold
    
    # Calculate total control effort (integral of moment magnitude)
    control_effort = np.sum(np.linalg.norm(magnetic_moments, axis=1))
    
    # Get final angular velocity magnitude
    final_omega_mag = omega_mags[-1]
    
    return settling_time, control_effort, final_omega_mag

def optimize_pid_gains(verbose: bool = True) -> dict:
    """
    Brute force search for optimal PID gains for bdot control.
    
    Args:
        verbose: Whether to print progress
    
    Returns:
        dict: Best parameters and their performance metrics
    """
    # Initial conditions (same as simulate_bdot)
    q = np.array([1, 0, 0, 0])
    Ω = np.array([0.1, -0.15, 0.08])
    x0 = np.concatenate((q, Ω))
    inertia = np.array([0.01461922201, 0.0412768466, 0.03235309961])
    mu_max = 3.0e-2
    dt = 5
    t_end = 3 * 60 * 60  # 3 hours (reduced from 24 for optimization speed)
    num_points = int(t_end // dt)
    t_arr = np.arange(num_points) * dt
    
    # Define gain ranges to test
    kp_range = np.logspace(0, 8, 9)  # [1e0, ..., 1e8]
    ki_range = np.logspace(0, 8, 9)  # [1e0, ..., 1e8]
    kd_range = np.logspace(0, 8, 9)  # [1e0, ..., 1e8]
    
    # Storage for best results
    best_performance = float('inf')  # Using weighted sum of metrics
    best_params = None
    best_metrics = None
    
    total_combinations = len(kp_range) * len(ki_range) * len(kd_range)
    combination_count = 0
    
    if verbose:
        print(f"Testing {total_combinations} gain combinations...")
    
    # Test all combinations
    for kp in kp_range:
        for ki in ki_range:
            for kd in kd_range:
                combination_count += 1
                if verbose:
                    print(f"\rProgress: {combination_count}/{total_combinations}", end="")
                
                # Initialize simulation arrays
                omegas = np.zeros((num_points, 3))
                magnetic_moments = np.zeros((num_points, 3))
                torques = np.zeros((num_points, 3))
                state_history = []
                
                # Run simulation with current gains
                x = x0.copy()
                for i in range(num_points):
                    t = t_arr[i]
                    B = magnetic_field(t)
                    
                    # Compute control with current gains
                    tau, mu = bdot_pid(x, B, mu_max, kp, ki, kd, state_history, dt)
                    
                    # Store values
                    omegas[i] = x[4:]
                    magnetic_moments[i] = mu
                    torques[i] = tau
                    
                    # Propagate dynamics
                    f = lambda t, x: quaternion_dynamics(x, dt, inertia, tau)
                    sol = solve_ivp(f, [t, t+dt], x, method='RK45')
                    x = sol.y[:, -1]
                
                # Evaluate performance
                settling_time, control_effort, final_omega = evaluate_performance(omegas, magnetic_moments)
                
                # Calculate weighted performance metric
                # Lower is better
                performance = (settling_time / num_points) * 0.5 + \
                            (control_effort / (mu_max * num_points)) * 0.3 + \
                            (final_omega / np.linalg.norm(Ω)) * 0.2
                
                # Update best if current is better
                if performance < best_performance:
                    best_performance = performance
                    best_params = {
                        'Kp': kp,
                        'Ki': ki,
                        'Kd': kd
                    }
                    best_metrics = {
                        'settling_time': settling_time * dt,  # Convert to seconds
                        'control_effort': control_effort,
                        'final_omega': final_omega,
                        'performance_score': performance
                    }
    
    if verbose:
        print("\n\nOptimization complete!")
        print("\nBest parameters found:")
        print(f"Kp: {best_params['Kp']:.2e}")
        print(f"Ki: {best_params['Ki']:.2e}")
        print(f"Kd: {best_params['Kd']:.2e}")
        print("\nPerformance metrics:")
        print(f"Settling time: {best_metrics['settling_time']:.1f} seconds")
        print(f"Control effort: {best_metrics['control_effort']:.2e}")
        print(f"Final omega magnitude: {best_metrics['final_omega']:.2e} rad/s")
        print(f"Overall performance score: {best_metrics['performance_score']:.3f}")
    
    return {'parameters': best_params, 'metrics': best_metrics}

def simulate_with_optimal_gains():
    """Run simulation using optimized gains"""
    # Get optimal gains
    optimal = optimize_pid_gains(verbose=True)
    
    # Run regular simulation with optimal gains
    print("\nRunning full simulation with optimal gains...")
    
    # Modify simulation call to use optimal gains
    # You'll need to modify your simulate_bdot() to accept PID gains as parameters
    # Then call it like:
    # simulate_bdot(Kp=optimal['parameters']['Kp'],
    #              Ki=optimal['parameters']['Ki'],
    #              Kd=optimal['parameters']['Kd'])

if __name__ == "__main__":
    # simulate_control_torques()
    # simulate_bdot()
    simulate_with_optimal_gains()