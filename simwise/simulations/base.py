import os
import copy
import datetime
from multiprocessing import Pool, cpu_count


import numpy as np
from tqdm import tqdm

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.data_structures.satellite_state import SatelliteState, interpolate_state
from simwise.math.coordinate_transforms import coe_to_mee
from simwise.navigation.attitude_ekf import ekf_measurement_update, ekf_time_update
from simwise.navigation.triad import triad
from simwise.math.quaternion import error_quaternion, regularize_quaternion, dcm_to_quaternion

def init_state(params):
    state = SatelliteState()

    # Initial orbit conditions
    state.orbit_keplerian = np.array([
        params.a, params.e, params.i,
        params.Ω, params.ω, params.θ
    ])
    state.orbit_mee = coe_to_mee(state.orbit_keplerian)

    # Initial time conditions
    state.jd = params.epoch_jd

    state.update_other_orbital_state_representations(params)
    state.update_environment(params)

    # Target attitude
    state.compute_target_attitude(params)

    # Initial attitude conditions
    state.q = state.q_d
    state.w = state.w_d
    state.control_mode = params.control_mode
    state.allocation_mode = params.allocation_mode

    # Initial filter conditions
    state.x_k = np.hstack((state.q, state.w))
    state.P_k = params.P
    state.Q = params.Q
    state.R = params.R    

    return state

def run_one(params):
    """This is the main function that runs the sim for a single set of params
    """
    state = init_state(params)
    infrequent_state_next = copy.deepcopy(state)

    states = []
    times = []
    num_points_attitude = int((params.t_end - params.t_start) // params.dt_attitude) + 1
    num_points_orbit = int((params.t_end - params.t_start) // params.dt_orbit) + 1

    for i in range(num_points_attitude):
        # Propagate orbit for greater time step - orbit
        if i % int(params.dt_orbit / params.dt_attitude) == 0:
            # Save the current state for linear interpolation in the future
            infrequent_state_prev = copy.deepcopy(infrequent_state_next)
            # TODO organize this
            state.orbit_mee = infrequent_state_prev.orbit_mee
            state.orbit_keplerian = infrequent_state_prev.orbit_keplerian
            state.v_mag_eci = infrequent_state_prev.v_mag_eci
            state.atmospheric_density = infrequent_state_prev.atmospheric_density
            state.v_sun_eci = infrequent_state_prev.v_sun_eci

            # Propagate orbit for greater time step - orbit
            infrequent_state_next.propagate_orbit(params)
            infrequent_state_next.update_other_orbital_state_representations(params)
            infrequent_state_next.update_environment(params)
            infrequent_state_next.propagate_time(params, params.dt_orbit)

        else: 
            # Interpolate things that take a long time to compute
            state.orbit_mee = interpolate_state(infrequent_state_prev, infrequent_state_next, state.t, attribute="orbit_mee")
            state.orbit_keplerian = interpolate_state(infrequent_state_prev, infrequent_state_next, state.t, attribute="orbit_keplerian")
            state.orbit_keplerian[5] = state.orbit_keplerian[5] % (2 * np.pi)
            state.v_mag_eci = interpolate_state(infrequent_state_prev, infrequent_state_next, state.t, attribute="v_mag_eci")
            state.atmospheric_density = interpolate_state(infrequent_state_prev, infrequent_state_next, state.t, attribute="atmospheric_density")
            state.v_sun_eci = interpolate_state(infrequent_state_prev, infrequent_state_next, state.t, attribute="v_sun_eci")   
    
        # Update other state representations
        state.update_other_orbital_state_representations(params)

        # Compute the target attitude
        state.compute_target_attitude(params)

        # Compute desired control torque
        state.compute_control_torque(params)

        # Allocate torque to actuators, and apply actuator noise model
        state.allocate_control_torque(params)

        # Propagate attitude at every step - smaller timestep
        state.torque_applied = state.control_torque + np.random.normal(0, params.noise_torque, 3)
        state.propagate_attitude(params)
        
        # Update forces
        state.update_forces(params)

        # Perform attitude determination
        if params.attitude_determination_mode == "EKF":
            state.x_k_minus, state.P_k_minus = ekf_time_update(state.x_k, state.P_k, state.Q, params.dt_attitude, params.inertia, state.control_torque)
            # state.x_k = state.x_k_minus
            # state.P_k = state.P_k_minus

            state.update_measurements(params)
            state.v_sun_eci = state.v_sun_eci / np.linalg.norm(state.v_sun_eci)
            state.x_k, state.P_k = ekf_measurement_update(state.x_k_minus, state.P_k_minus, state.R, state.v_sun_meas, state.v_mag_meas, state.v_sun_eci, state.v_mag_eci)
            state.x_k[:4] = regularize_quaternion(state.x_k[:4])
        elif params.attitude_determination_mode == "TRIAD":
            state.update_measurements(params)
            R = triad(state.v_sun_eci, state.v_mag_eci, state.v_sun_meas, state.v_mag_meas)
            state.x_k[:4] = dcm_to_quaternion(R)
        else:
            raise ValueError("Invalid attitude determination mode")

        # Define time in terms of smaller timestep - attitude
        state.propagate_time(params, params.dt_attitude)

        state.attitude_knowedge_error = regularize_quaternion(error_quaternion(state.x_k[:4], state.q))

        states.append(copy.deepcopy(state))
        times.append(state.t)

    return states, times

def run_orbit(params):
    """This function runs only the orbit simulation
    """
    state = init_state(params)

    states = []
    times = []
    num_points_orbit = int((params.t_end - params.t_start) // params.dt_orbit) + 1

    for i in range(num_points_orbit):
        state.propagate_time(params, params.dt_attitude)
        state.propagate_orbit(params)
        states.append(copy.deepcopy(state))
        times.append(state.t)

    return states, times

def run_attitude(params):
    """This function runs only the attitude simulation
    """
    state = init_state(params)

    states = []
    times = []
    num_points_attitude = int((params.t_end - params.t_start) // params.dt_attitude) + 1

    for _ in range(num_points_attitude):
        state.propagate_time(params, params.dt_attitude)
        times.append(state.t)
        state.compute_control_torque(params)
        state.allocate_control_torque(params)
        state.propagate_attitude(params)
        states.append(copy.deepcopy(state))

    return states, times

def run_dispersions(params, runner=run_one):
    # Generate 3 dispersed instances
    dispersed_instances = params.generate_dispersions(params.num_dispersions)

    # Print all values that are dispersed
    for dispersed_params in dispersed_instances:
        for attr in dir(dispersed_params):
            if not attr.startswith("_") and isinstance(getattr(dispersed_params, attr), (ArrayParameter, QuaternionParameter, ScalarParameter)):
                # print(attr, getattr(dispersed_params, attr))
                pass
    
    num_workers = cpu_count()
    states_from_dispersions = []
    times_from_dispersions = []
    
    with Pool(processes=num_workers) as pool:
        with tqdm(total=params.num_dispersions, desc=f"Simulating {params.num_dispersions} runs") as pbar:
            for states, times in pool.imap_unordered(runner, dispersed_instances):
                states_from_dispersions.append(states)
                times_from_dispersions.append(times)
                pbar.update()
    
    sim_data_dir = "data"
    if not os.path.exists(sim_data_dir):
        os.makedirs(sim_data_dir)
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    np.save(f"sim_data_dir/states_{curr_time}.npy", np.array(states_from_dispersions))
    np.save(f"sim_data_dir/times_{curr_time}.npy", np.array(times_from_dispersions))
    return np.array(states_from_dispersions), np.array(times_from_dispersions)
