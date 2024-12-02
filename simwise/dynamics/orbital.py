import numpy as np
from simwise.constants import *

def mee_dynamics(elements, mu, dt, f_perturbation):
    """ Do rigid body dyamics of a body in MEE

    Args:
        elements (_type_): _description_
        mu (_type_): _description_
        dt (_type_): _description_
        f (_type_): _description_
    """

    p = elements[0]
    f = elements[1]
    g = elements[2]
    h = elements[3]
    k = elements[4]
    L = elements[5]

    cosL = np.cos(L)
    sinL = np.sin(L)
    w = 1 + f*cosL+g*sinL
    root_p_µ = np.sqrt(p/mu)
    s_2 = 1 + h**2 + k**2

    # calculate matrix
    A = np.array([
        [0,     2*(p/w),               0                     ],
        [sinL,  (1/w)*((w+1.)*cosL+f), -(g/w)*(h*sinL-k*cosL)],
        [-cosL, ((w+1.)*sinL+g)/w,     (f/w)*(h*sinL+k*cosL) ],
        [0,     0,                     (s_2*cosL)/(2*w)      ],
        [0,     0,                     (s_2*sinL)/(2*w)      ],
        [0,     0,                     ((h*sinL-k*cosL)/w)   ]
    ])

    A = root_p_µ * A
    b = [0, 0, 0, 0, 0, np.sqrt(mu/p**3)*w**2]

    return A @ f_perturbation + b




def get_altitude(rv, radius_earth=6371):
    """
    Calculate altitude from state vector
    
    Parameters:
    rv : array-like
        State vector [x, y, z, vx, vy, vz]
    radius_earth : float, optional
        Radius of the Earth in km (default is 6371 km)
    
    Returns:
    float
        Altitude in km
    """
    r = rv[:3]
    return np.linalg.norm(r) - radius_earth

def get_velocity(rv):
    """
    Calculate velocity magnitude from state vector
    
    Parameters:
    rv : array-like
        State vector [x, y, z, vx, vy, vz]
    
    Returns:
    float
        Velocity magnitude in km/s
    """
    v = rv[3:]
    return np.linalg.norm(v)