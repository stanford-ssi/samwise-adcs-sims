import numpy as np
from typing import Callable


# Function adpated from Alfonso Gonazelez YouTube channel
# Runge Kutta solver for the differential equation
def RK4(fn: Callable[[np.ndarray[float]], np.ndarray[float]], y: float, h: float):
    k1 = fn(y)
    k2 = fn(y + 0.5 * k1 * h)
    k3 = fn(y + 0.5 * k2 * h)
    k4 = fn(y + k3 * h)

    return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
