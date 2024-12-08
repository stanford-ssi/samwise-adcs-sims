import copy
from multiprocessing import Pool, cpu_count


import numpy as np
from tqdm import tqdm

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.data_structures.satellite_state import SatelliteState, interpolate_state
from simwise.math.coordinate_transforms import coe_to_mee

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

    # Initial attitude conditions
    state.q = params.q_initial
    state.w = params.w_initial
    state.q_d = params.q_desired
    state.w_d = params.w_desired
    state.control_mode = params.control_mode
    state.allocation_mode = params.allocation_mode

    return state

def run_one(params):
    """This is the main function that runs the sim for a single set of params
    """
    state = init_state(params)

    states = []
    times = []
    num_points_attitude = int((params.t_end - params.t_start) // params.dt_attitude) + 1
    num_points_orbit = int((params.t_end - params.t_start) // params.dt_orbit) + 1
    
    # Initialize the state of the orbit
    # TODO make an initialization function
    state.update_other_orbital_state_representations(params)
    state.update_environment(params)
    infrequent_state_next = copy.deepcopy(state)

    for i in range(num_points_attitude):
        
        # Propagate orbit for greater time step - orbit
        if i % int(params.dt_orbit / params.dt_attitude) == 0:
            # Save the current state for linear interpolation in the future
            infrequent_state_prev = copy.deepcopy(infrequent_state_next)
            # TODO organize this
            state.orbit_mee = infrequent_state_prev.orbit_mee
            state.orbit_keplerian = infrequent_state_prev.orbit_keplerian
            state.magnetic_field = infrequent_state_prev.magnetic_field
            state.atmospheric_density = infrequent_state_prev.atmospheric_density
            state.r_sun_eci = infrequent_state_prev.r_sun_eci

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
            state.magnetic_field = interpolate_state(infrequent_state_prev, infrequent_state_next, state.t, attribute="magnetic_field")
            state.atmospheric_density = interpolate_state(infrequent_state_prev, infrequent_state_next, state.t, attribute="atmospheric_density")
            state.r_sun_eci = interpolate_state(infrequent_state_prev, infrequent_state_next, state.t, attribute="r_sun_eci")   
    
        # Update other state representations
        state.update_other_orbital_state_representations(params)

        # Compute the target attitude
        state.compute_target_attitude(params)

        # Compute desired control torque
        state.compute_control_torque(params)

        # Allocate torque to actuators, and apply actuator noise model
        state.allocate_control_torque(params)

        # Propagate attitude at every step - smaller timestep
        state.propagate_attitude(params)
        
        # Update forces
        state.update_forces(params)

        # Define time in terms of smaller timestep - attitude
        state.propagate_time(params, params.dt_attitude)

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
    
    return np.array(states_from_dispersions), np.array(times_from_dispersions)
