from datetime import datetime, timedelta
import math
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple

from .bodies import Body
from .dataclasses import ClassicalOrbitalElements
from .differential_equations.two_body import ODE as TwoBodyODE
from .solvers.rk4 import RK4
from .utils import get_gmst_from_epoch


class Orbit:
    # Default epoch: Vernal Equinox 2024 (March 20, 06:00 UTC)
    DEFAULT_EPOCH = datetime(2024, 3, 20, 6, 0, 0)

    def __init__(self, state: np.ndarray, body: Body, epoch: datetime = DEFAULT_EPOCH):
        """
        Initialize the Orbit object using a state vector and a celestial body.

        :param state: State vector (position and velocity) as a NumPy array.
        :param body: An instance of the Body class representing the celestial body.
        :param epoch: Initializing time of orbit.
        """
        self.body = body
        self.time = epoch
        self.state = state

    @classmethod
    def from_coes(
        cls,
        coes: ClassicalOrbitalElements,
        body: Body,
        epoch: datetime = DEFAULT_EPOCH,
    ):
        """
        Initialize the Orbit object using classical orbital elements and a celestial body.

        :param coes: Tuple of classical orbital elements (sma, ecc, inc, raan, aop, ta).
        :param body: An instance of the Body class representing the celestial body.
        :param epoch: Initializing time of orbit.
        :return: An instance of the Orbit class.
        """

        state = cls.Coes2State(coes, body.gravitational_parameter)
        return cls(state, body, epoch)

    @classmethod
    def from_state(cls, state: np.ndarray, body: Body, epoch: datetime = DEFAULT_EPOCH):
        """
        Initialize the Orbit object using a state vector and a celestial body.

        :param state: State vector (position and velocity) as a NumPy array.
        :param body: An instance of the Body class representing the celestial body.
        :param epoch: Initializing time of orbit.
        :return: An instance of the Orbit class.
        """
        return cls(state, body, epoch)

        # Convert Classical Orbital Elements to State Vector

    @staticmethod
    def Coes2State(coes: ClassicalOrbitalElements, mu: float) -> np.ndarray[float]:
        # calculate orbital angular momentum of satellite
        h = math.sqrt(mu * (coes.sma * (1 - coes.ecc**2)))
        # h = 1.6041e13

        cos_ta = math.cos(math.radians(coes.ta))
        sin_ta = math.sin(math.radians(coes.ta))

        r_w = h**2 / mu / (1 + coes.ecc * cos_ta) * np.array((cos_ta, sin_ta, 0))
        v_w = mu / h * np.array((-sin_ta, coes.ecc + cos_ta, 0))

        # rotate to inertian frame
        R = Rotation.from_euler("ZXZ", [-coes.aop, -coes.inc, -coes.raan], degrees=True)
        r_rot = r_w @ R.as_matrix()
        v_rot = v_w @ R.as_matrix()

        return np.concatenate((r_rot, v_rot))

    @staticmethod
    def State2Coes(
        state: np.ndarray[float], mu: float
    ) -> Tuple[float, float, float, float, float, float]:
        r_vec = state[:3]
        v_vec = state[3:]

        # Position and velocity magnitudes
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        v_r = np.dot(r_vec / r, v_vec)
        v_p = np.sqrt(v**2 - v_r**2)

        # Orbital angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        sma = h

        # Inclination
        inc = np.arccos(h_vec[2] / h)

        # RAAN
        K = np.array((0, 0, 1))
        N_vec = np.cross(K, h_vec)
        N = np.linalg.norm(N_vec)
        raan = 2 * np.pi - np.arccos(N_vec[0] / N)

        # Eccentricity
        e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
        ecc = np.linalg.norm(e_vec)

        # AOP
        aop = 2 * np.pi - np.arccos(np.dot(N_vec, e_vec) / (N * ecc))

        # True anomaly
        ta = np.arccos(np.dot(r_vec / r, e_vec / ecc))

        return ClassicalOrbitalElements(sma, ecc, inc, ta, aop, raan)

        # Transform Earth-Centered Inertial to Earth-Centered Earth-Fixed

    @staticmethod
    def Eci2Ecef(state: np.ndarray[float], t: float) -> np.ndarray[float]:
        # omega = 0.261799387799149  # radians/hour
        # theta = Cal2Gmst()
        # theta = float((omega * t / 60 / 60) % (2 * math.pi))
        theta = get_gmst_from_epoch(t)
        R = Rotation.from_euler("Z", theta, degrees=False)
        return state @ R.as_matrix()

    # Transform Earth-Centered Earth-Fixed to Geocentric coordinates
    @staticmethod
    def Ecef2Geoc(state: np.ndarray[float], r: float):
        geoc = np.zeros(3)
        geoc[0] = math.degrees(math.atan2(state[1], state[0]))  # longitude
        geoc[1] = math.degrees(math.asin(state[2] / np.linalg.norm(state)))  # latitude
        geoc[2] = np.linalg.norm(state) - r  # altitude
        return geoc

    def get_state_eci(self) -> np.ndarray[float]:
        """
        :return: earth-centered inertial coordinates [x, y, z, vx, vy, vz]
        """
        return self.state

    def get_state_ecef(self, t: float) -> np.ndarray[float]:
        """
        :param t: time in seconds since epoch
        :return: earth-centered earth-fixed coordinates [x, y, z]
        """
        state = self.state[:3]
        return self.Eci2Ecef(state, t)

    def get_state_geoc(self, t: float) -> np.ndarray[float]:
        """
        :param t: time in seconds since epoch
        :return: geocentric coordinates [longitude, latitude, altitude]
        """
        return self.Ecef2Geoc(self.get_state_ecef(t), self.body.radius)

    def propagate(self, tspan: int, dt: int) -> np.ndarray[float]:
        """
        Propagate the orbit to a future time.

        :param tspan: Duration of the simulation.
        :param dt: Time step for the numerical integration.
        :return: An array of state vectors at each time step.
        """

        steps = int(abs(tspan) / dt)
        signedDt = math.copysign(dt, tspan)

        newStates = np.zeros((steps, 6))
        newStatesGeoc = np.zeros((steps, 3))

        diffEqn = lambda state: TwoBodyODE(state, self.body.gravitational_parameter)

        for i in range(steps):
            newStates[i] = RK4(diffEqn, self.state, signedDt)
            newStatesGeoc[i] = self.get_state_geoc(self.time.timestamp() + i*signedDt)
            self.state = newStates[i]
            self.time += timedelta(seconds=signedDt)


        return newStates, newStatesGeoc
