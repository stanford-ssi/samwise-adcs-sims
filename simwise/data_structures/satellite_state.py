"""
This file defines the main SatelliteState class, which is the
root of all dunamics updating
"""
import numpy as np

from scipy.integrate import solve_ivp

from simwise.math.quaternion import *

from simwise.control.attitude_control import *
from simwise.control.bdot import *

from simwise.dynamics.orbital import *
from simwise.dynamics.attitude import *
from simwise.data_structures.parameters import Parameters
from simwise.math.coordinate_transforms import *
from simwise.math.frame_transforms import *
import simwise.constants
from simwise.forces.drag import dragPertubationTorque
from simwise.world_model.magnetic_field import magnetic_field



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
    r_eci: np.ndarray  # [x, y, z] in [m]
    v_eci: np.ndarray  # [vx, vy, vz] in [m/s]
    r_ecef: np.ndarray  # [x, y, z] in [m]
    lla_wgs84: np.ndarray  # [lat, lon, alt] in [rad, rad, m]

    #  Magnetorquer data
    B: np.ndarray   # [T]
    mu: np.ndarray  # [A • m^2]
    
    # Inertial Variables:
    # Velocity Components of Satellite on Orbit
    v_vec_trn: np.ndarray # [m/s]   shape(,n)   - V vector magnitude in TRN
    # Radial Components of Satellite Position on Orbit

    # TODO: replace this with r_eci
    r_vec: np.ndarray     # [m]  shape(3,n)     - R position vector - {x,y,z}
    # Satellite altitude (wrt non-J2 Earth)
    h: np.ndarray         # [m]
    # Euler Angles
    e_angles: np.ndarray  # [radians]
    
    
    # Pertubation Forces:
    Drag: np.ndarray   # [N]
    

    def propagate_time(self, params: Parameters, dt):
        """Update this state's time"""
        self.t = self.t + dt
        self.jd = self.jd + dt / constants.SECONDS_PER_DAY

    ###———————————————————————————————————————————————————————————————————————###
    ###                      BEGIN Flight Computer Model                      ###
    ###———————————————————————————————————————————————————————————————————————###
    def compute_control_torque(self, params: Parameters):
        """Compute the desired control torque for the current state\
            
        """
        x_attitude = np.hstack((self.q, self.w))
        x_attitude_desired = np.hstack((self.q_d, self.w_d))

        self.error_angle = quaternions_to_axis_angle(self.q, self.q_d)[0]
        if self.control_mode == "PD":
            self.desired_control_torque = compute_control_torque(x_attitude,
                                                     x_attitude_desired,
                                                     params.K_p,
                                                     params.K_d,
                                                     tau_max=params.max_torque)
        elif self.control_mode == "BDOT":
            tau, mu = bdot_bang_bang(x_attitude, self.B, params.mu_max)
            self.desired_control_torque = tau
            self.mu = mu
            # tau, mu = bdot_step_bang_bang(x_attitude, self.B, params.mu_max)
            # tau, mu = bdot_proportional(x_attitude, self.B, params.mu_max)
        else:
            raise ValueError(f"Unknown control mode '{self.control_mode}'")

    def allocate_control_torque(self, params: Parameters):
        #TODO: Implement a control allocation algorithm
        if self.allocation_mode == "MagicActuators":
            self.control_torque = self.desired_control_torque
        else:
            raise ValueError(f"Unknown control allocation mode '{self.allocation_mode}'")
        
    ###———————————————————————————————————————————————————————————————————————###
    ###                       END Flight Computer Model                       ###
    ###———————————————————————————————————————————————————————————————————————###

    ###———————————————————————————————————————————————————————————————————————###
    ###                       BEGIN Ground Truth Dynamics Model               ###
    ###———————————————————————————————————————————————————————————————————————###
    def propagate_attitude(self, params: Parameters):
        """Update this state to represent advancing attitude"""
        x_attitude = np.hstack((self.q, self.w))

        def attitude_ode(t, x):
            return attitude_dynamics(x_attitude, params.dt_attitude, params.inertia, self.control_torque)

        sol = solve_ivp(
            attitude_ode,
            [self.t, self.t + params.dt_attitude],
            x_attitude,
            method='RK45'
        )

        x_attitude_new = sol.y[:, -1]

        # Derive Attitude State Information:
        self.q = x_attitude_new[:4]                                 # Quaternion
        self.w = x_attitude_new[4:]                                 # Angular Velocity
        self.e_angles = quaternion_to_euler(self.q, sequence="zyx")    # Euler Angles

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

        # Solve for the MEE and Keplerian Orbital States
        self.orbit_mee = sol.y[:, -1]
        self.orbit_mee[5] = self.orbit_mee[5] % (2 * np.pi)
        self.orbit_keplerian = mee_to_coe(self.orbit_mee)
        self.orbit_keplerian[5] = self.orbit_keplerian[5] % (2 * np.pi)
        
        # Solve for Velocity, Position and Altitude at this Orbital Time Step
        self.v_vec_trn = get_velocity(coe_to_rv(mee_to_coe(self.orbit_mee)))
        self.h = get_altitude(coe_to_rv(mee_to_coe(self.orbit_mee)))

    def update_other_state_representations(self, params: Parameters):
        rv_eci = coe_to_rv(self.orbit_keplerian)
        self.r_eci = rv_eci[:3]
        self.v_eci = rv_eci[3:]
        self.r_ecef = ECI_to_ECEF_tabular(self.r_eci, params.ecef_pn_table, self.jd)
        self.lla_wgs84 = ECEF_to_topocentric(self.r_ecef)

        # XXX This is a hack to get the magnetic field only when we propagate orbit
        self.magnetic_field = magnetic_field(self.lla_wgs84, self.jd)

    def update_environment(self, params):
        ''' Solve directly for pertubation torques in one call'''
        
        # Solve for the Drag:
        self.Drag = dragPertubationTorque(params, self.e_angles, self.v_vec_trn, self.h)
        
    ###———————————————————————————————————————————————————————————————————————###
    ###                       END Ground Truth Dynamics Model                 ###
    ###———————————————————————————————————————————————————————————————————————###
        