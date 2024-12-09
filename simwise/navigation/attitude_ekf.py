import numpy as np

from simwise.math.quaternion import rotate_vector_by_quaternion, quaternion_to_dcm

def stm(omega, J_inertia, dt):
    """Propagate quaternion and angular velocity."""
    omega_mag = np.linalg.norm(omega)
    omega = omega
    if omega_mag > 1e-6:
        Ω = np.array([
            [0,        -omega[0],   -omega[1],  -omega[2]],
            [omega[0], 0,           omega[2],   -omega[1]],
            [omega[1], -omega[2],   0,          omega[0]],
            [omega[2], omega[1],    -omega[0],  0]
        ])
        A_qq = np.eye(4) + 0.5 * Ω * dt
    else:
        A_qq = np.eye(4)

    A_ww = np.eye(3)  # Simple integration for angular velocities
    A = np.block([[A_qq, np.zeros((4, 3))], [np.zeros((3, 4)), A_ww]])
    return A


def ekf_time_update(x, P, Q, dt, J_inertia, tau_torque):
    """EKF time update (prediction)."""
    omega = x[4:]
    Φ = stm(omega, J_inertia, dt)
    
    # Torque effects on angular velocity
    omega_dot = tau_torque / J_inertia
    x_k_minus = Φ @ x
    x_k_minus[4:] += omega_dot * dt # Integrate angular velocity

    # Normalize quaternion
    x_k_minus[:4] /= np.linalg.norm(x_k_minus[:4])

    P_k_minus = Φ @ P @ Φ.T + Q
    return x_k_minus, P_k_minus


def ekf_measurement_update(x_k_minus, P_k_minus, R, v_sun_meas, v_mag_meas, v_sun_eci, v_mag_eci):
    """EKF measurement update."""
    q_k_minus = x_k_minus[:4]
    z = np.hstack((v_sun_meas, v_mag_meas))

    # Predicted measurements
    v_sun_predict = rotate_vector_by_quaternion(v_sun_eci, q_k_minus)
    v_mag_predict = rotate_vector_by_quaternion(v_mag_eci, q_k_minus)

    # Innovation
    h = np.hstack((v_sun_predict, v_mag_predict))
    y = z - h

    # Measurement Jacobian H
    q0, q1, q2, q3 = q_k_minus
    S = v_sun_eci
    M = v_mag_eci

    # Compute partial derivatives of R wrt q0,q1,q2,q3:
    # see quaternion.py quaternion to DCM conversion
    dR_dq0 = np.array([
        [ 2*q0,    -2*q3,    2*q2    ],
        [ 2*q3,     2*q0,   -2*q1    ],
        [-2*q2,     2*q1,    2*q0    ]
    ])
    dR_dq1 = np.array([
        [ 2*q1,     2*q2,    2*q3    ],
        [ 2*q2,    -2*q1,   -2*q0    ],
        [ 2*q3,     2*q0,   -2*q1    ]
    ])
    dR_dq2 = np.array([
        [-2*q2,     2*q1,    2*q0    ],
        [ 2*q1,     2*q2,    2*q3    ],
        [-2*q0,     2*q3,   -2*q2    ]
    ])
    dR_dq3 = np.array([
        [-2*q3,    -2*q0,    2*q1    ],
        [ 2*q0,    -2*q3,    2*q2    ],
        [ 2*q1,     2*q2,    2*q3    ]
    ])

    # For sun measurement
    dhsun_dq0 = dR_dq0 @ S
    dhsun_dq1 = dR_dq1 @ S
    dhsun_dq2 = dR_dq2 @ S
    dhsun_dq3 = dR_dq3 @ S

    # For mag measurement
    dhmag_dq0 = dR_dq0 @ M
    dhmag_dq1 = dR_dq1 @ M
    dhmag_dq2 = dR_dq2 @ M
    dhmag_dq3 = dR_dq3 @ M

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

    # Columns for angular velocities remain zero since h doesn't depend on them

    # Kalman gain
    S = H @ P_k_minus @ H.T + R
    K = P_k_minus @ H.T @ np.linalg.inv(S)

    # Update state and covariance
    x_k = x_k_minus + K @ y

    # Joseph formulation guarantees positive semi-definite covariance
    term1 = (np.eye(len(P_k_minus)) - K @ H)
    P_k = term1 @ P_k_minus @ term1.T + K @ R @ K.T
    
    # Normalize quaternion
    x_k[:4] /= np.linalg.norm(x_k[:4])
    
    return x_k, P_k
