''' bdot algorithms in order of simplest (and most robust) to fanciest '''
import numpy as np

def bdot_bang_bang(x: np.ndarray, B: np.ndarray, mu_max: float = 2.5e-2, deadband: float = 1e-10) -> tuple:
    """
    B-dot bang-bang control law with independent magnetorquer coils.
    
    Args:
        x (np.ndarray): State vector [q1, q2, q3, q4, ω1, ω2, ω3]
        B (np.ndarray): Magnetic field vector in body frame [T]
        mu_max (float): Maximum magnetic moment per axis [A⋅m²]
    
    Returns:
        tuple: (Control torque vector [N⋅m], Magnetic moment vector [A⋅m²])
    """
    omega = x[4:]  # Extract angular velocity from state vector
    
    # Calculate apparent rate of change of B field
    dBdt = -np.cross(omega, B)
    
    # Bang-bang control
    mu = np.zeros(3)
    for i in range(3):
        if abs(dBdt[i]) < deadband:
            mu[i] = 0
        else:
            mu[i] = -mu_max if dBdt[i] > 0 else mu_max
    
    # Compute resulting torque
    tau = np.cross(mu, B)
    
    return tau, mu

def bdot_step_bang_bang(x: np.ndarray, B: np.ndarray, mu_max: float = 2.5e-2, K: float = 1e5) -> tuple:
    """
    Step bang-bang control using proportional B-dot control as reference.
    
    Args:
        x (np.ndarray): State vector [q1, q2, q3, q4, ω1, ω2, ω3]
        B (np.ndarray): Magnetic field vector in body frame [T]
        mu_max (float): Maximum magnetic moment per axis [A⋅m²]
        K (float): Proportional gain for reference
    
    Returns:
        tuple: (Control torque vector [N⋅m], Magnetic moment vector [A⋅m²])
    """
    omega = x[4:]  # Extract angular velocity from state vector
    
    # Calculate apparent rate of change of B field
    dBdt = -np.cross(omega, B)
    
    # Get proportional control reference
    mu_prop = -K * dBdt
    mu_prop_max = np.max(np.abs(mu_prop))  # Get maximum magnitude for normalization
    
    # Step bang-bang control
    mu = np.zeros(3)
    for i in range(3):
        if mu_prop_max < 1e-10:  # Avoid division by zero
            mu[i] = 0
            continue
            
        prop_ratio = abs(mu_prop[i]) / mu_prop_max
        
        if prop_ratio < 0.33:
            mu[i] = 0
        elif prop_ratio > 0.66:
            mu[i] = np.sign(mu_prop[i]) * mu_max
        else:
            mu[i] = np.sign(mu_prop[i]) * 0.5 * mu_max
    
    # Compute resulting torque
    tau = np.cross(mu, B)
    
    return tau, mu

def bdot_proportional(x: np.ndarray, B: np.ndarray, mu_max: float = 2.5e-2, K: float = 1e5) -> tuple:
    """
    B-dot proportional control law.
    
    Args:
        x (np.ndarray): State vector [q1, q2, q3, q4, ω1, ω2, ω3]
        B (np.ndarray): Magnetic field vector in body frame [T]
        mu_max (float): Maximum magnetic moment per axis [A⋅m²]
        K (float): Control gain
    
    Returns:
        tuple: (Control torque vector [N⋅m], Magnetic moment vector [A⋅m²])
    """
    omega = x[4:]  # Extract angular velocity from state vector
    
    # Calculate apparent rate of change of B field
    dBdt = -np.cross(omega, B)
    
    # Proportional control law
    mu = -K * dBdt
    
    # Saturate magnetic moment
    for i in range(3):
        if abs(mu[i]) > mu_max:
            mu[i] = np.sign(mu[i]) * mu_max
    
    # Compute resulting torque
    tau = np.cross(mu, B)
    
    return tau, mu


def bdot_pid(x: np.ndarray, B: np.ndarray, mu_max: float = 2.5e-2, 
           Kp: float = 1e5, Ki: float = 1e2, Kd: float = 1e3,
           state_history: list = [], dt: float = 0.1) -> tuple:
    """
    B-dot PID control law.
    
    Args:
        x (np.ndarray): State vector [q1, q2, q3, q4, ω1, ω2, ω3]
        B (np.ndarray): Magnetic field vector in body frame [T]
        mu_max (float): Maximum magnetic moment per axis [A⋅m²]
        Kp (float): Proportional gain
        Ki (float): Integral gain
        Kd (float): Derivative gain
        state_history (list): List of past states [(x, B, dBdt)] for derivative/integral
        dt (float): Time step for integration/derivative [s]
    
    Returns:
        tuple: (Control torque vector [N⋅m], Magnetic moment vector [A⋅m²])
    """
    omega = x[4:]  # Extract angular velocity from state vector
    
    # Calculate apparent rate of change of B field
    dBdt = -np.cross(omega, B)
    
    # Update state history with current dBdt
    state_history.append((x.copy(), B.copy(), dBdt.copy()))
    if len(state_history) > 100:  # Limit history size
        state_history.pop(0)
        
    # Initialize control terms
    mu = np.zeros(3)
    
    # Proportional term (basic bdot)
    p_term = -Kp * dBdt
    
    # Integral term (if enabled)
    i_term = np.zeros(3)
    if Ki != 0 and len(state_history) > 1:
        # Integrate dBdt over history
        for past_state in state_history[-10:]:  # Use last 10 points
            past_dBdt = past_state[2]
            i_term += -Ki * past_dBdt * dt
            
    # Derivative term (if enabled)
    d_term = np.zeros(3)
    if Kd != 0 and len(state_history) > 1:
        # Get rate of change of dBdt
        prev_dBdt = state_history[-2][2]
        d2Bdt2 = (dBdt - prev_dBdt) / dt
        d_term = -Kd * d2Bdt2
    
    # Sum all terms
    mu = p_term + i_term + d_term
    
    # Saturate magnetic moment
    for i in range(3):
        if abs(mu[i]) > mu_max:
            mu[i] = np.sign(mu[i]) * mu_max
    
    # Compute resulting torque
    tau = np.cross(mu, B)
    
    return tau, mu