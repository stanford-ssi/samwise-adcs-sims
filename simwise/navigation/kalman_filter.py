import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

class KalmanFilterQuaternion:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        
        # Initial state: quaternion (identity), angular velocity (zero)
        self.x = np.zeros(7)  # [q1, q2, q3, q4, wx, wy, wz]
        self.x[:4] = np.array([1, 0, 0, 0])  # Identity quaternion
        self.P = np.eye(7)  # Initial state covariance
    
    def predict(self):
        # Extract quaternion and angular velocity
        q = self.x[:4]
        omega = self.x[4:]
        
        # Quaternion kinematics
        omega_quat = np.hstack(([0], omega))
        dq_dt = 0.5 * R.from_quat(q).as_matrix() @ omega_quat
        
        # Discrete-time propagation
        q_new = q + dq_dt * self.dt
        q_new = normalize_quaternion(q_new)
        
        # Angular velocity (assuming constant)
        omega_new = omega
        
        # Update state
        self.x[:4] = q_new
        self.x[4:] = omega_new
        
        # Update covariance
        F = np.eye(7)  # State transition matrix approximation
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, magnetic_field, sun_measurements, reference_field, sun_directions):
        # Compute expected measurements
        q = R.from_quat(self.x[:4])
        h_m = q.apply(reference_field)  # Expected magnetic field
        h_s = np.array([q.apply(d) for d in sun_directions])  # Expected sun sensor readings
        
        # Assemble measurement
        h = np.hstack((h_m, h_s.ravel()))
        z = np.hstack((magnetic_field, sun_measurements.ravel()))
        
        # Measurement Jacobian (approximation)
        H = np.eye(len(z), 7)  # Placeholder for a real Jacobian
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x += K @ (z - h)
        self.x[:4] = normalize_quaternion(self.x[:4])  # Re-normalize quaternion
        
        # Update covariance
        self.P = (np.eye(7) - K @ H) @ self.P


def quaternion_to_rotation_jacobian(q, vector):
    """
    Computes the Jacobian of the rotated vector with respect to quaternion.
    """
    q0, q1, q2, q3 = q
    vx, vy, vz = vector
    jac = np.array([
        [2*q0*vx - 2*q3*vy + 2*q2*vz, 2*q1*vx + 2*q2*vy + 2*q3*vz, -4*q2*vx + 2*q1*vy - 2*q0*vz, -4*q3*vx - 2*q0*vy + 2*q1*vz],
        [2*q3*vx + 2*q0*vy - 2*q1*vz, 2*q2*vx - 4*q1*vy - 2*q0*vz, 2*q1*vx + 2*q0*vy + 2*q3*vz, -4*q3*vy + 2*q2*vz + 2*q0*vx],
        [-2*q2*vx + 2*q1*vy + 2*q0*vz, 2*q3*vx + 2*q0*vy + 2*q1*vz, 2*q2*vx - 4*q0*vz - 2*q1*vy, 2*q3*vx - 4*q0*vz + 2*q2*vy]
    ])
    return jac

def compute_jacobian(q, ref_field, sun_dirs):
    """
    Compute the full measurement Jacobian matrix.
    """
    num_sensors = len(sun_dirs)
    H = np.zeros((3 + num_sensors, 7))  # 3 magnetic, N sun sensors, 7 states (quaternion + omega)
    
    # Magnetic field Jacobian
    H[:3, :4] = quaternion_to_rotation_jacobian(q, ref_field)
    
    # Sun sensor Jacobians
    for i, n in enumerate(sun_dirs):
        if np.dot(n, ref_field) > 0:  # Sun sensor active condition
            H[3 + i, :4] = n @ quaternion_to_rotation_jacobian(q, ref_field)
    
    return H
