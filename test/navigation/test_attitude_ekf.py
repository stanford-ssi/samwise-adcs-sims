import numpy as np

from simwise.navigation.attitude_ekf import ekf_measurement_update, ekf_time_update, stm
from simwise.math.quaternion import quaternion_to_euler
from simwise.utils.plots import plot_subplots

def random_quaternion():
    """Generate a random unit quaternion."""
    q = np.random.randn(4)
    return q / np.linalg.norm(q)

def test_stm():
    # Set a time step
    e_angles = []
    times = []
    dt = 0.1

    # Define a constant angular velocity about x-axis
    omega = np.array([0.0, 0.0, 0.1])  # rad/s around x-axis
    
    # Assume a simple diagonal inertia
    J_inertia = np.array([1.0, 1.0, 1.0])
    
    # Initial state: quaternion + angular velocity
    # Start with identity quaternion [1,0,0,0]
    q_init = np.array([1.0, 0.0, 0.0, 0.0])
    x_init = np.hstack((q_init, omega))
    
    # Apply the STM a few times to see how the state evolves
    x = x_init.copy()
    print("Initial state:", x)
    
    for i in range(100):
        A = stm(x[:4], omega, J_inertia, dt)
        x = A @ x
        # Normalize quaternion to avoid drift due to linearization approximations
        x[:4] /= np.linalg.norm(x[:4])
        times.append(i*dt)
        e_angles.append(quaternion_to_euler(x[:4]))

    times = np.array(times)
    e_angles = np.array(e_angles)
    plot_subplots(times, e_angles, ["Roll", "Pitch", "Yaw"], "Time[s]", "Euler Angles vs Time")
    

def test_ekf():
    # ----------------------
    # Parameters
    # ----------------------
    dt = 0.1  # time step (s)
    J_inertia = np.array([1, 1, 1])  # scalar inertia for simplicity
    tau_torque = np.array([0.0, 0.0, 0.0])  # no external torque

    # Process noise covariance Q: small uncertainty in both orientation and angular velocity
    Q = np.diag([1e-7, 1e-7, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6])

    # Measurement noise covariance R: set to some small noise
    R_cov = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])

    # Initial state: quaternion + angular velocity
    q_init = random_quaternion()
    omega_init = np.array([0.01, 0.02, -0.01])  # some small angular velocity
    x = np.hstack((q_init, omega_init))

    # Initial covariance
    P = np.diag([1e-4]*4 + [1e-3]*3)

    # "True" state for generating synthetic measurements
    q_true = q_init.copy()
    omega_true = omega_init.copy()

    # Define known ECI vectors (e.g., Sun and magnetometer reference vectors)
    v_sun_eci = np.array([1.0, 0.0, 0.0])   # Suppose sun vector is along x
    v_mag_eci = np.array([0.0, 1.0, 0.0])   # Suppose magnetic field is along y

    # Synthetic true measurements: rotate known ECI vectors by the true quaternion
    from simwise.math.quaternion import rotate_vector_by_quaternion
    v_sun_meas_true = rotate_vector_by_quaternion(v_sun_eci, q_true)
    v_mag_meas_true = rotate_vector_by_quaternion(v_mag_eci, q_true)

    # Add noise to measurements
    measurement_noise = np.random.randn(6) * 1e-2  # Some random noise
    v_sun_meas = v_sun_meas_true + measurement_noise[0:3]
    v_mag_meas = v_mag_meas_true + measurement_noise[3:6]

    # ----------------------
    # Run EKF Time Update (Prediction)
    # ----------------------
    x_pred, P_pred = ekf_time_update(x, P, Q, dt, J_inertia, tau_torque)
    print("Predicted State:", x_pred)
    print("Predicted Covariance:\n", P_pred)

    # ----------------------
    # Run EKF Measurement Update
    # ----------------------
    x_upd, P_upd = ekf_measurement_update(x_pred, P_pred, R_cov, v_sun_meas, v_mag_meas, v_sun_eci, v_mag_eci)
    print("\nUpdated State:", x_upd)
    print("Updated Covariance:\n", P_upd)

    raise Exception
