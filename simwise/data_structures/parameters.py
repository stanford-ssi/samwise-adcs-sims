import numpy as np

class Parameters:
    dt = 1/60 # [sec]
    t_end = 2 * 60 # [sec]

    inertia = np.array([0.01461922201, 0.0412768466, 0.03235309961]) # [kg m^2]

    # Controls
    K_p = 0.0005
    K_d = 0.005

    max_torque = 0.0032 # [Nm] 
    noise_torque = 0.00000288 # [Nm]

    mu_max = 2*3.0e-2  # 0.030 A⋅m²