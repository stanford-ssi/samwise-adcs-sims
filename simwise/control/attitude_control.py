from simwise.math.quaternion import *
import numpy as np

def compute_control_torque(x, x_desired, K_p=1, K_d=1, tau_max=None):
    """
    Compute torque from current to target state using P-D control.
    Can also specify a maximum actuator torque, `tau_max`, to limit
    the norm of the appliec torque
    """
    # x = [q, w]
    # x_d = [q_d, w_d]
    
    q = normalize_quaternion(x[:4])
    q_d = normalize_quaternion(x_desired[:4])
    theta, rotation_vector = quaternions_to_axis_angle(q, q_d)
    
    w = x[4:]
    w_d = x_desired[4:]
    w_error = w_d - w

    tau = K_p * theta * rotation_vector + K_d * w_error

    if tau_max is not None:
        if np.linalg.norm(tau) > tau_max:
            tau = tau_max * tau / np.linalg.norm(tau) 

    return tau

