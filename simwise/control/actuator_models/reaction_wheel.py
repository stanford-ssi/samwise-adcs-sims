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
    Assuming a tetrahedral arrangement of reaction wheels, the direction vectors are:
    [0, 2*np.sqrt(2)/3, 2/np.sqrt(3),           -2/np.sqrt(3)],
    [0, 0,              np.sqrt(2)/np.sqrt(3),  -np.sqrt(2)/np.sqrt(3)],
    [1, -1/3,           -1/3,                   -1/3]
    """

    # raction wheel inertia
    motor_MOI = 7.90e-07 # kg*m^2

    # reaction wheel angular velocity noise

    W = np.array([
        [0, 2*np.sqrt(2)/3, -2/np.sqrt(3),           -2/np.sqrt(3)],
        [0, 0,              np.sqrt(2)/np.sqrt(3),  -np.sqrt(2)/np.sqrt(3)],
        [1, -1/3,           -1/3,                   -1/3]
    ])
    
    W_pseudoinv = W.T@np.linalg.inv(W@W.T)

    # reaction wheel torque
    reaction_wheel_torque = W_pseudoinv@tau # assume N*m

    # reaction wheel speed
    reaction_wheel_speed = reaction_wheel_torque/motor_MOI # rad/s
    
    # Compute control torques end here
    
    # Now we backtrack to get the modelled torque

    # Calculate speed based on adding in Gaussian noise
    # This is for the simulation only, not for the actual reaction wheel speed
    actual_reaction_wheel_speed = reaction_wheel_speed+np.random.normal(0, abs(reaction_wheel_speed*0.025))

    # actual torque
    actual_torque = actual_reaction_wheel_speed*motor_MOI

    return actual_torque
