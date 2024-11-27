"""
This file defines the main SatelliteState class, which is the
root of all dunamics updating
"""
import numpy as np

from scipy.integrate import solve_ivp

from simwise.math.quaternion import *

from simwise.attitude.attitude_control import *
from simwise.attitude.bdot import *

from simwise.orbit.equinoctial import *
from simwise.data_structures.parameters import Parameters
import simwise.constants


class SatelliteState:
    # Time
    t: float = 0  # [sec]

    # Attitude
    q: np.ndarray
    w: np.ndarray  # [rad / sec]

    # Desired State
    q_d: np.ndarray
    w_d: np.ndarray  # [rad / sec]

    # Attitude info
    control_torque: np.ndarray  # [Nm]
    error_angle: float  # [rad]

    # Orbital elements in multiple forms
    orbit_keplerian: np.ndarray  # [a, e, i, Ω, ω, θ] in [m, rad]
    orbit_mee: np.ndarray  # [p, f, g, h, k, L]

    #  Magnetorquer data
    B: np.ndarray   # [T]
    mu: np.ndarray  # [A • m^2]
    
    # Pertubation Forces:
    Drag: np.ndarray   # [N]
    

    def propagate_time(self, params: Parameters, dt):
        """Update this state's time"""
        # Use this only for attitude
        self.t = self.t + dt

    def propagate_attitude_control(self, params: Parameters):
        """Update this state to represent advancing attitude"""
        x_attitude = np.hstack((self.q, self.w))
        x_attitude_desired = np.hstack((self.q_d, self.w_d))

        self.error_angle = angle_axis_between(self.q, self.q_d)[0]
        self.control_torque = compute_control_torque(x_attitude,
                                                     x_attitude_desired,
                                                     params.K_p,
                                                     params.K_d,
                                                     tau_max=params.max_torque)

        def attitude_ode(t, x):
            return attitude_dynamics(x_attitude, params.dt_attitude, params.inertia, self.control_torque)

        sol = solve_ivp(
            attitude_ode,
            [self.t, self.t + params.dt_attitude],
            x_attitude,
            method='RK45'
        )

        x_attitude_new = sol.y[:, -1]

        self.q = x_attitude_new[:4]
        self.w = x_attitude_new[4:]

    def propagate_attitude_bdot(self, params: Parameters):
        """Update this state to reflect one cycle of B-dot detumbling"""
        x_attitude = np.hstack((self.q, self.w))

        tau, mu = bdot_bang_bang(x_attitude, self.B, params.mu_max)
        # tau, mu = bdot_step_bang_bang(x_attitude, self.B, params.mu_max)
        # tau, mu = bdot_proportional(x_attitude, self.B, params.mu_max)

        tau, mu = bdot_pid(x_attitude, self.B, params.mu_max)

        self.control_torque = tau
        self.mu = mu

        # Propagate dynamics
        def attitude_ode(t, x):
            return attitude_dynamics(x, params.dt_attitude, params.inertia, self.control_torque)

        sol = solve_ivp(
            attitude_ode,
            [self.t, self.t + params.dt_attitude],
            x_attitude,
            method='RK45'
        )

        x_attitude_new = sol.y[:, -1]
        self.q = x_attitude_new[:4]
        self.w = x_attitude_new[4:]

    def propagate_orbit(self, params: Parameters):
        """Update this state to represent advancing orbit"""
        f_perturbation = np.array([0, 0, 0])

        def orbit_ode(t, mee):
            return mee_dynamics(mee, simwise.constants.MU_EARTH, params.dt_orbit, f_perturbation)

        sol = solve_ivp(
            orbit_ode,
            [self.t, self.t + params.dt_orbit],
            self.orbit_mee,
            method='RK45'
        )

        self.orbit_mee = sol.y[:, -1]
        self.orbit_keplerian = mee2coe(self.orbit_mee)
