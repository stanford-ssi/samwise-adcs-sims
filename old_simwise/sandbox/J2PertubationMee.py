def mee_dynamics_with_j2(mee, mu, re, j2):
    p, f, g, h, k, L = mee
    
    # Calculate auxiliary variables
    s2 = 1 + h**2 + k**2
    w = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / w
    
    # Calculate classical orbital elements
    e = np.sqrt(f**2 + g**2)
    i = np.arctan2(2 * np.sqrt(h**2 + k**2), 1 - h**2 - k**2)
    
    # Calculate J2 acceleration in RTN frame
    sin_i = np.sin(i)
    cos_i = np.cos(i)
    sin_2i = np.sin(2*i)
    sin_L = np.sin(L)
    cos_L = np.cos(L)
    
    a_J2_R = -1.5 * j2 * mu * re**2 / r**4 * (1 - 3 * sin_i**2 * sin_L**2)
    a_J2_T = -1.5 * j2 * mu * re**2 / r**4 * (sin_i**2 * np.sin(2*L))
    a_J2_N = -1.5 * j2 * mu * re**2 / r**4 * (sin_2i * sin_L)
    
    # Convert J2 acceleration to inertial frame
    cos_u = (f + cos_L) / (1 + e * cos_L)
    sin_u = (g + sin_L) / (1 + e * cos_L)
    
    a_J2_I = np.array([
        a_J2_R * cos_u - a_J2_T * sin_u,
        a_J2_R * sin_u + a_J2_T * cos_u,
        a_J2_N
    ])
    
    # Calculate the transformation matrix A
    sqrt_p_over_mu = np.sqrt(p / mu)
    A = np.array([
        [0, 2*p/w * sqrt_p_over_mu, 0],
        [sqrt_p_over_mu * sin_L, sqrt_p_over_mu/w * ((w+1)*cos_L + f), -sqrt_p_over_mu * (g/w) * (h*sin_L - k*cos_L)],
        [-sqrt_p_over_mu * cos_L, sqrt_p_over_mu/w * ((w+1)*sin_L + g), sqrt_p_over_mu * (f/w) * (h*sin_L - k*cos_L)],
        [0, 0, sqrt_p_over_mu * s2 * cos_L/(2*w)],
        [0, 0, sqrt_p_over_mu * s2 * sin_L/(2*w)],
        [0, 0, sqrt_p_over_mu/w * (h*sin_L - k*cos_L)]
    ])

    # Calculate the unperturbed rates
    n = np.sqrt(mu / p**3)
    b = np.array([0, 0, 0, 0, 0, n * w**2 / p])

    # Calculate the perturbed rates
    mee_rates = A @ a_J2_I + b

    return mee_rates