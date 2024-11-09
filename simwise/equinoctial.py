import numpy as np

def coe2mee(elements):
    """Transform classic (keplerian) orbital elements to modified equinoctial elements.
sdajkshdkjsah
    Args:
        elements (np.ndarray): orbital elements of form [a, e, i, Ω, ω, θ]
    """
    a = elements[0]
    e = elements[1]
    i = elements[2]
    Ω = elements[3]
    ω = elements[4]
    θ = elements[5]

    p = a * (1 - e**2)
    f = e * np.cos(ω + θ)
    g = e * np.sin(ω + θ)
    h = np.tan(i / 2) * np.cos(Ω)
    k = np.tan(i / 2) * np.sin(Ω)
    L = Ω + ω + θ
    return np.array([p, f, g, h, k, L])

def mee2coe(elements):
    """_summary_

    Args:
        elements (_type_): _description_
    """
    p = elements[0]
    f = elements[1]
    g = elements[2]
    h = elements[3]
    k = elements[4]
    L = elements[5]

    a = p / (1 - f**2 - g**2)
    e = np.sqrt(f**2 + g**2)
    i = np.arctan2(2 * np.sqrt(h**2 + k**2), 1 - h**2 - k**2)
    Ω = np.arctan2(k, h)
    ω = np.arctan2(g, f) - L
    θ = L - Ω - ω
    return np.array([a, e, i, Ω, ω, θ])

def mee2cartesian(elements, mu):
    """_summary_

    Args:
        elements (_type_): _description_
        mu (_type_): _description_
    """
    raise NotImplementedError

def mee_dynamics(elements, mu, dt, f_perturbation):
    """_summary_

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

    a = p / (1 - f**2 - g**2)
    e = np.sqrt(f**2 + g**2)
    i = np.arctan2(2 * np.sqrt(h**2 + k**2), 1 - h**2 - k**2)
    Ω = np.arctan2(k, h)
    ω = np.arctan2(g, f) - L
    θ = L - Ω - ω
    s = (1+h**2+k**2)**0.5
    
    A = np.array([
        [0, ((2*p)/ω)*(p/mu)**0.5, 0],
        [((p/mu)**0.5)*np.sin(L), ((p/mu)**0.5)*(1/ω)*((ω+1)*np.cos(L) + f), -(p/mu)**0.5*(g/ω)*(h*np.sin(L) - k*np.cos(L))],
        [-((p/mu)**0.5)*np.cos(L), ((p/mu)**0.5)*(1/ω)*((ω+1)*np.sin(L)+g), ((p/mu)**0.5)*(f/ω)*(h*np.sin(L) - k*np.cos(L))],
        [0, 0, ((p/mu)**0.5)*(s**2*np.cos(L))/(2*ω)],
        [0, 0, ((p/mu)**0.5)*(s**2*np.sin(L))/(2*ω)],
        [0, 0, ((p/mu)**0.5)*(1/ω)*(h*np.sin(L) - k*np.cos(L))]
    ])

    b = np.array([
        0,
        0,
        0,
        0,
        0,
        np.sqrt(mu*p)*(ω/p)**2
    ])
    print(A, f)

    return A @ f_perturbation + b