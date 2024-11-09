import numpy as np
from scipy.integrate import solve_ivp
from quaternion import quaternion2euler, quaternion_dynamics, compute_control_torque, angle_axis_between, compute_control_torque_lqr
from graph_utils import graph_euler, graph_vector_matplotlib, graph_quaternion, graph_quaternion_matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    # Initial conditions
    q = np.array([1, 0, 0, 0])
    Ω = np.array([0.1, 0.1, 0.1]) # [rad/s]
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

    # LQR weighting matrices
    Q=np.diag([1,1,1,1,1,1,1])
    R=np.diag([1,1,1])

    # Noise parameter (standard deviation of gaussian) [Nm]
    tau_noise = 0.00000288

    # Maximum actuator torque [Nm]
    tau_max = 0.0032

    dt = 1 # [sec]
    t_end = 5 * 60 # [sec] --> minutes
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

        f_lqr = lambda t, x: quaternion_dynamics(x, dt, inertia, compute_control_torque_lqr(x, x_d, Q, R, inertia, tau_max=tau_max), noise=tau_noise)
        f_pid = lambda t, x: quaternion_dynamics(x, dt, inertia, compute_control_torque(x, x_d, K_p, K_d, tau_max=tau_max), noise=tau_noise)
        sol = solve_ivp(f_pid, [t, t+dt], x, method='RK45')
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
