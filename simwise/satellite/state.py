import numpy as np
from simwise.math.quaternion import Quaternion

class SatelliteState:
    def __init__(self, q_eci2body: Quaternion, w_eci: np.array, r_eci: np.array, v_eci: np.array, t: float = 0.0, mjd_epoch: float = 0.0):
        self.q = q_eci2body     # scalar-last, ECI to body frame
        self.w = w_eci          # [rad/s] in ECI frame
        self.r = r_eci          # [m] in ECI frame
        self.v = v_eci          # [m/s] in ECI frame
        self.t = t              # [s]

        # Epoch
        self.mjd_epoch = mjd_epoch # [days]

    def __repr__(self):
        return f"SatelliteState(t={self.t}, q={self.q}, w={self.w}, r={self.r}, v={self.v})"

    def __str__(self):
        return f"SatelliteState(t={self.t}, q={self.q}, w={self.w}, r={self.r}, v={self.v})"

    def __add__(self, other):
        return SatelliteState(self.q + other.q, self.w + other.w, self.r + other.r, self.v + other.v, self.t)

    def __sub__(self, other):
        return SatelliteState(self.q - other.q, self.w - other.w, self.r - other.r, self.v - other.v, self.t)

    def __mul__(self, scalar):
        return SatelliteState(self.q * scalar, self.w * scalar, self.r * scalar, self.v * scalar, self.t * scalar, self.mjd_epoch)
    
    def __truediv__(self, scalar):
        return SatelliteState(self.q / scalar, self.w / scalar, self.r / scalar, self.v / scalar, self.t / scalar, self.mjd_epoch)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __rtruediv__(self, scalar):
        return self.__truediv__(scalar)