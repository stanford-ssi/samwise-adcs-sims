import numpy as np
from simwise.satellite.state import SatelliteState
from simwise.satellite.params import SatelliteParams
from simwise.world.sun import sun_vector_eci


class Satellite:
    """
    Bundles a satellite's truth state, physical parameters, and optional
    estimation algorithm.

    Force models (torques, perturbations) are properties of the simulation
    environment — pass them to propagate_satellite(), not here.

    Sensors are fixed: magnetometer, sun sensors, IMU, GPS.

    Estimator protocol (duck-typed):
        .update(measurements: dict, dt: float) -> None
        .state: SatelliteState  (current estimate)
    """

    def __init__(
        self,
        state: SatelliteState,
        params: SatelliteParams,
        estimator=None,
    ):
        self.state = state
        self.params = params
        self.estimator = estimator
        self.history: list[SatelliteState] = [state]
        self.est_history: list[SatelliteState] = []

    def __repr__(self):
        return f"Satellite(t={self.state.t:.1f}s, estimator={self.estimator is not None})"

    def read_sensors(self) -> dict:
        """Returns a dict of measurements from all onboard sensors."""
        return {
            'imu': self._read_imu(),
            'magnetometer': self._read_magnetometer(),
            'sun_sensors': self._read_sun_sensors(),
            'gps': self._read_gps(),
        }

    def _read_gps(self) -> dict:
        r_noise = np.random.normal(0, 10.0, 3)   # ~10 m 1-sigma
        v_noise = np.random.normal(0, 0.1, 3)    # ~0.1 m/s 1-sigma
        return {
            'r': self.state.r + r_noise,
            'v': self.state.v + v_noise,
        }
    
    def _read_sun_sensors(self) -> dict: 
        q_eci2body = self.state.q
        sun_eci = sun_vector_eci(self.state)
        sun_body = q_eci2body.rot(sun_eci)

        # add gaussian noise
        noise = np.random.normal(0, 0.01, 3)
        sun_body += noise

        return {'sun_body': sun_body}
    
    def _read_magnetometer(self) -> dict:
        r_eci = self.state.r
        b_eci = b_field_dipole(r_eci)
        q_eci2body = self.state.q
        b_body = q_eci2body.rot(b_eci)

        # add gaussian noise
        noise = np.random.normal(0, 0.01, 3)
        b_body += noise

        return {'b_body': b_body}
    
    def _read_imu(self) -> dict: 
        w_eci = self.state.w
        w_body = self.state.q.rot(w_eci)

        # add noise and gyro random walk bias to measurement
        noise = np.random.normal(0, 0.01, 3)   # ~0.01 rad/s 1-sigma
        w_body += noise
        w_body += self.state.gyro_bias

        return {'w_body': w_body}