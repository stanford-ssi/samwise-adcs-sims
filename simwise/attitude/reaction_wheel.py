'''Simulation of reaction wheels'''

import numpy as np
from simwise.attitude.attitude_control import compute_control_torque

def torque_to_actual_torque(x, x_desired, K_p=1, K_d=1, tau_max=None):
    """
    torque->reaction wheel speed change->add Gaussian noise->actual torque
    input: tau - calculated torque
    output a_tau - actual torque added the Gaussian noise to raction wheel speed
    """

    tau = compute_control_torque(x, x_desired, K_p=1, K_d=1, tau_max=None)

    # Psudoinverse matrix of four-wheel pyramid reaction wheel
    """
    ai, bi: direction vector of reaction wheels #1 and #2
    ci, di: direction vector of reaction wheels #3 and #4
    """
    a1 = 1/np.sqrt(2) 
    b1 = 1/np.sqrt(2) 
    a2 = 1/np.sqrt(2)
    b2 = 1/np.sqrt(2)
    c3 = 1/np.sqrt(2)
    d3 = 1/np.sqrt(2)
    c4 = 1/np.sqrt(2)
    d4 = 1/np.sqrt(2)

    # raction wheel inertia
    motor_MOI = 7.90e-07 # kg*m^2

    # reaction wheel angular velocity noise

    W = np.array([[a1, -a2, 0, 0],
                  [b1, b2, c3, c4],
                  [0, 0, d3, -d4]])
    
    W_pseudoinv = W.T@np.linalg.inv(W@W.T)

    reaction_wheel_torque = W_pseudoinv@tau # assume N*m

    reaction_wheel_speed = reaction_wheel_torque/motor_MOI # rad/s

    actual_reaction_wheel_speed = reaction_wheel_speed+np.random.normal(0, abs(reaction_wheel_speed*0.025))

    actual_torque = actual_reaction_wheel_speed*motor_MOI

    return actual_torque
