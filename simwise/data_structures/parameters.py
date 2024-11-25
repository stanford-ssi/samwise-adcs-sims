import numpy as np

class Parameters:
    dt_orbit = 1           # [sec] (1 sec)
    dt_attitude = 0.1       # [sec] 
    t_start = 0             # [sec]
    t_end = 1 * 60         # [sec] (1 minute)

    inertia = np.array([0.01461922201, 0.0412768466, 0.03235309961]) # [kg m^2]

    # Controls
    K_p = 0.0005
    K_d = 0.005

    max_torque = 0.0032 # [Nm] 
    noise_torque = 0.00000288 # [Nm]

    mu_max = 2*3.0e-2  # 0.030 A⋅m²
    
    # Initial Orbit Properties:
    a = 7000e3 # [m]
    e = 0.001
    i = 0.1 # [rad]
    Ω = 0.1 # [rad]
    ω = 0.1 # [rad]
    θ = 0.1 # [rad]
    initial_orbit_state = np.array([a, e, i, Ω, ω, θ])
    
    # Attitude initial conditions
    q_initial = np.array([1, 0, 0, 0])
    w_initial = np.array([0.0, 0.2, 0.1])  # [rad/s]
    q_desired = np.array([0.5, 0.5, 0.5, 0.5])
    w_desired = np.array([0, 0, 0])  # [rad/s]

    
    
    
    
    
    
    