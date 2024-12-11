import numpy as np
from simwise.math.quaternion import rotate_vector_by_quaternion, quaternion_to_dcm

def stm(q, omega, J_inertia, dt):
    """Propagate quaternion and angular velocity."""
    J1, J2, J3 = J_inertia
    q0, q1, q2, q3 = q
    omega_mag = np.linalg.norm(omega)
    if omega_mag > 0:
        Ω = np.array([
            [0,        -omega[0],   -omega[1],  -omega[2]],
            [omega[0], 0,            omega[2],   -omega[1]],
            [omega[1], -omega[2],    0,           omega[0]],
            [omega[2], omega[1],    -omega[0],    0]
        ])
        A_qq = np.cos(omega_mag*dt/2)*np.eye(4) + np.sin(omega_mag*dt/2)/omega_mag*Ω # + (1-np.cos(omega_mag*dt))/omega_mag**2*Ω@Ω
    else:
        print("Small omega")
        A_qq = np.eye(4)

    A_ww = np.array([
        [1, (J2-J3)/J1*omega[2], (J2-J3)/J1*omega[1]],
        [(J3-J1)/J2*omega[2], 1, (J3-J1)/J2*omega[0]],
        [(J1-J2)/J3*omega[1], (J1-J2)/J3*omega[0], 1]
    ])

    # A_qw = -0.5*dt*np.array([
    #     [-q1, -q2, -q3],
    #     [q0, -q3,  q2],
    #     [q3,  q0, -q1],
    #     [-q2,  q1,  q0]
    # ])

    A = np.block([[A_qq, np.zeros((4, 3))],
                  [np.zeros((3, 4)), A_ww]])
    return A


def ekf_time_update(x, P, Q, dt, J_inertia, tau_torque):
    """EKF time update (prediction)."""
    q = x[:4]
    omega = x[4:]
    Φ = stm(q, omega, J_inertia, dt)

    # Compute angular acceleration
    if J_inertia.shape == (3,):
        omega_dot = tau_torque / J_inertia
    else:
        # If J_inertia is a matrix
        omega_dot = np.linalg.inv(J_inertia) @ tau_torque

    # Propagate state
    x_k_minus = Φ @ x
    x_k_minus[4:] += omega_dot * dt  # integrate angular velocity

    # Normalize quaternion
    x_k_minus[:4] /= np.linalg.norm(x_k_minus[:4])

    # Propagate covariance
    P_k_minus = Φ @ P @ Φ.T + Q
    return x_k_minus, P_k_minus


def ekf_measurement_update(x_k_minus, P_k_minus, R_cov, v_sun_meas, v_mag_meas, v_sun_eci, v_mag_eci):
    """EKF measurement update."""
    q_k_minus = x_k_minus[:4]
    z = np.hstack((v_sun_meas, v_mag_meas))

    # Predicted measurements
    v_sun_predict = rotate_vector_by_quaternion(v_sun_eci, q_k_minus)
    v_mag_predict = rotate_vector_by_quaternion(v_mag_eci, q_k_minus)

    # Innovation
    h = np.hstack((v_sun_predict, v_mag_predict))
    y = z - h
    print(y)
    q0, q1, q2, q3 = q_k_minus

    # Correct partial derivatives of rotation matrix w.r.t. quaternions
    # DCM from quaternion q = (q0, q1, q2, q3):
    # R[0,0] = q0^2 + q1^2 - q2^2 - q3^2, etc.
    # For brevity, we trust these corrected derivatives:

    dR_dq0 = np.array([
        [ 2*q0,    2*q3,    -2*q2],
        [-2*q3,    2*q0,     2*q1],
        [ 2*q2,   -2*q1,     2*q0]
    ])
    dR_dq1 = np.array([
        [ 2*q1,    2*q2,     2*q3],
        [ 2*q2,   -2*q1,     2*q0],
        [ 2*q3,   -2*q0,    -2*q1]
    ])
    dR_dq2 = np.array([
        [-2*q2,    2*q1,    -2*q0],
        [ 2*q1,    2*q2,     2*q3],
        [ 2*q0,    2*q3,    -2*q2]
    ])
    dR_dq3 = np.array([
        [-2*q3,    2*q0,     2*q1],
        [-2*q0,   -2*q3,     2*q2],
        [ 2*q1,    2*q2,     2*q3]
    ])

    # For sun measurement
    dhsun_dq0 = dR_dq0 @ v_sun_eci
    dhsun_dq1 = dR_dq1 @ v_sun_eci
    dhsun_dq2 = dR_dq2 @ v_sun_eci
    dhsun_dq3 = dR_dq3 @ v_sun_eci

    # For mag measurement
    dhmag_dq0 = dR_dq0 @ v_mag_eci
    dhmag_dq1 = dR_dq1 @ v_mag_eci
    dhmag_dq2 = dR_dq2 @ v_mag_eci
    dhmag_dq3 = dR_dq3 @ v_mag_eci

    # Construct H
    H = np.zeros((6,7))
    H[0:3,0] = dhsun_dq0
    H[0:3,1] = dhsun_dq1
    H[0:3,2] = dhsun_dq2
    H[0:3,3] = dhsun_dq3

    H[3:6,0] = dhmag_dq0
    H[3:6,1] = dhmag_dq1
    H[3:6,2] = dhmag_dq2
    H[3:6,3] = dhmag_dq3

    # Angular velocity columns remain zeros, as measurement does not depend on them

    # Compute Kalman gain
    S_cov = H @ P_k_minus @ H.T + R_cov
    K = P_k_minus @ H.T @ np.linalg.inv(S_cov)

    # Update state and covariance
    x_k = x_k_minus + K @ y
    I = np.eye(len(P_k_minus))
    P_k = (I - K @ H) @ P_k_minus @ (I - K @ H).T + K @ R_cov @ K.T
    P_k = 0.5 * (P_k + P_k.T)

    # Normalize quaternion again after update
    x_k[:4] /= np.linalg.norm(x_k[:4])

    return x_k, P_k
