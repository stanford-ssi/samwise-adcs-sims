import copy
from multiprocessing import Pool, cpu_count


import numpy as np
from tqdm import tqdm

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.orbit.equinoctial import coe2mee

def init_state(params):
    state = SatelliteState()

    # Initial orbit conditions
    state.orbit_keplerian = np.array([
        params.a, params.e, params.i,
        params.Ω, params.ω, params.θ
    ])
    state.orbit_mee = coe2mee(state.orbit_keplerian)

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

    for i in range(num_points_attitude):
        # Define time in terms of smaller timestep - attitude
        state.propagate_time(params, params.dt_attitude)
        
        # Compute desired control torque
        state.compute_control_torque(params)

        # Allocate torque to actuators
        state.allocate_control_torque(params)

        # Propagate attitude at every step - smaller timestep
        state.propagate_attitude(params)
        
        # Propagate orbit for greater time step - orbit
        if i % int(params.dt_orbit / params.dt_attitude) == 0:
            state.propagate_orbit(params)
        
        # Calculate perturbation forces
        state.calculate_pertubation_forces(params)
        
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
        # t = params.t_start + i * params.dt_orbit
        # state.t = t
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
