"""
Simple orbit simulation without attitude control.
This simulation just propagates the satellite's orbit without any pointing requirements.
"""

import numpy as np
from simwise.simulations.base import run_one, run_dispersions
from simwise.data_structures.parameters import Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.utils.plots import plot_results
import simwise.constants as constants
from simwise.math.coordinate_transforms import *
from simwise.math.frame_transforms import *

def initialize_params():
    """Initialize simulation parameters with defaults."""
    params = Parameters()
    
    # Disable attitude control
    params.control_mode = "none"
    
    # Set simulation duration
    params.sim_time = 90 * 60  # 90 minutes (roughly one orbit)
    params.time_step = 1.0     # 1 second time step
    
    return params

def initialize_state(params):
    """Initialize satellite state with defaults."""
    state = SatelliteState()
    
    # Set initial time
    state.t = 0.0
    state.jd = 2459000.5  # Example Julian date (July 13, 2020)
    
    # Set initial attitude to identity quaternion (no rotation)
    state.q = np.array([1.0, 0.0, 0.0, 0.0])
    state.w = np.array([0.0, 0.0, 0.0])  # No initial angular velocity
    state.e_angles = np.array([0.0, 0.0, 0.0])  # No initial Euler angles
    
    # Initialize control-related variables
    state.q_d = np.array([1.0, 0.0, 0.0, 0.0])
    state.w_d = np.array([0.0, 0.0, 0.0])
    state.control_torque = np.zeros(3)
    state.error_angle = 0.0
    state.control_mode = "none"
    state.allocation_mode = "none"
    
    # Set up a circular orbit
    altitude = 500.0 * 1000  # 500 km in meters
    
    # Calculate orbital parameters for a circular orbit
    r_earth = constants.RADIUS_EARTH
    r_orbit = r_earth + altitude
    
    # Calculate orbital velocity for circular orbit
    v_orbit = np.sqrt(constants.MU_EARTH / r_orbit)
    
    # Initialize position and velocity in ECI frame (circular orbit in xy-plane)
    state.r_eci = np.array([r_orbit, 0.0, 0.0])
    state.v_eci = np.array([0.0, v_orbit, 0.0])
    
    # Convert to Keplerian elements
    a = r_orbit  # Semi-major axis for circular orbit
    e = 0.0      # Eccentricity (circular)
    i = 0.0      # Inclination (equatorial)
    Omega = 0.0  # Right ascension of ascending node
    omega = 0.0  # Argument of periapsis
    theta = 0.0  # True anomaly
    
    state.orbit_keplerian = np.array([a, e, i, Omega, omega, theta])
    
    # Convert to Modified Equinoctial Elements
    p = a * (1 - e**2)  # Semi-latus rectum
    f = e * np.cos(omega + Omega)
    g = e * np.sin(omega + Omega)
    h = np.tan(i/2) * np.cos(Omega)
    k = np.tan(i/2) * np.sin(Omega)
    L = Omega + omega + theta
    
    state.orbit_mee = np.array([p, f, g, h, k, L])
    
    # Initialize other required state variables
    state.r_ecef = ECI_to_ECEF_tabular(state.r_eci, params.ecef_pn_table, state.jd)
    state.lla_wgs84 = ECEF_to_topocentric(state.r_ecef)
    state.v_vec_trn = state.v_eci.copy()  # Simplified for now
    state.h = altitude
    state.Drag = np.zeros(3)
    
    # Initialize environment variables
    state.magnetic_field = np.zeros(3)
    state.atmospheric_density = 0.0
    state.r_sun_eci = np.array([1.0, 0.0, 0.0]) * constants.AU
    
    return state

def custom_run_one(params):
    """Run a single simulation without attitude control."""
    # Initialize state
    state = initialize_state(params)
    
    # Create arrays to store results
    num_steps = int(params.sim_time / params.time_step)
    states = []
    times = np.zeros(num_steps)
    
    # Run simulation
    for i in range(num_steps):
        # Store current state
        states.append(state.copy())
        times[i] = i * params.time_step
        
        # Update time
        state.propagate_time(params, params.time_step)
        
        # Update orbit
        state.propagate_orbit(params)
        state.update_other_orbital_state_representations(params)
        
        # Update environment (but skip forces that might cause NaN issues)
        state.update_environment(params)
        
        # Propagate attitude without control
        state.control_torque = np.zeros(3)  # No control torque
        state.propagate_attitude(params)
    
    return states, times

def run():
    """Run the simple orbit simulation."""
    print("Running simple orbit simulation without attitude control...")
    
    # Initialize parameters
    params = initialize_params()
    
    # Run simulation
    states, times = custom_run_one(params)
    
    # Plot results
    plot_results(states, times, params)
    
    print("Simulation complete!")
    
    return states, times

if __name__ == "__main__":
    run()