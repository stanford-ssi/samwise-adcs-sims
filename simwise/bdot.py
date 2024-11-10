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

def bdot_adaptive(x: np.ndarray, B: np.ndarray, mu_max: float = 2.5e-2,
                 state_history: list = [], window_size: int = 100) -> tuple:
    """
    Adaptive B-dot control that:
    1. Learns from effectiveness of past actions
    2. Adapts control parameters based on convergence rate
    3. Uses dynamic deadband based on noise estimation
    4. Implements intelligent multi-axis scheduling
    
    Args:
        x: State vector [q1, q2, q3, q4, ω1, ω2, ω3]
        B: Magnetic field vector [T]
        mu_max: Maximum magnetic moment [A⋅m²]
        state_history: List of past states for learning
        window_size: Size of historical window for adaptation
    """
    omega = x[4:]
    dBdt = -np.cross(omega, B)
    
    # Update state history
    state_history.append((x.copy(), B.copy()))
    if len(state_history) > window_size:
        state_history.pop(0)
    
    # Initialize outputs
    mu = np.zeros(3)
    
    if len(state_history) > 10:  # Need some history for adaptation
        # Calculate convergence rates for each axis
        convergence_rates = np.zeros(3)
        for i in range(3):
            past_omegas = [state[0][4+i] for state in state_history[-10:]]
            convergence_rates[i] = np.abs(np.mean(np.diff(past_omegas)))
        
        # Estimate noise levels from history
        noise_levels = np.zeros(3)
        for i in range(3):
            B_history = [state[1][i] for state in state_history]
            noise_levels[i] = np.std(np.diff(B_history))
        
        # Dynamic deadband based on noise
        deadbands = noise_levels * 3.0  # 3-sigma threshold
        
        # Prioritize axes based on convergence and effectiveness
        priorities = convergence_rates * np.abs(dBdt)
        best_axis = np.argmax(priorities)
        
        # Adaptive activation threshold
        if abs(dBdt[best_axis]) > deadbands[best_axis]:
            mu[best_axis] = -np.sign(dBdt[best_axis]) * mu_max
            
            # Optional: Coordinate secondary axis if highly effective
            second_best = np.argsort(priorities)[-2]
            if priorities[second_best] > 0.8 * priorities[best_axis]:
                mu[second_best] = -np.sign(dBdt[second_best]) * mu_max * 0.5
    
    else:  # Fall back to simple bang-bang when insufficient history
        best_axis = np.argmax(np.abs(dBdt))
        if abs(dBdt[best_axis]) > 1e-6:
            mu[best_axis] = -np.sign(dBdt[best_axis]) * mu_max
    
    tau = np.cross(mu, B)
    return tau, mu