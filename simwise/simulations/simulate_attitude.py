# Basic attitude simulation with no perturbation torques
import numpy as np
from tqdm import tqdm
import copy

from simwise.data_structures.parameters import Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.utils.plots import plot_states_plotly


def simulate_attitude(start_time, end_time, time_step, orbit_states=None):
    params = Parameters()
    state = SatelliteState()
    state.q = np.array([1, 0, 0, 0])
    state.w = np.array([0.0, 0.2, 0.1])  # [rad/s]
    state.q_d = np.array([0.5, 0.5, 0.5, 0.5])
    state.w_d = np.array([0, 0, 0])  # [rad/s]

    # Define parameters
    params.dt = time_step
    params.t_end = end_time - start_time

    # Simulate
    print("Simulating attitude...")
    states: list[SatelliteState] = []
    num_points = int(params.t_end // params.dt)

    for i in tqdm(range(num_points)):
        state.t = start_time + i * time_step
        state.propagate_time(params)
        state.propagate_attitude_control(params)

        # If orbit_states is provided, update state with orbital information
        if orbit_states is not None:
            state.position = orbit_states[i][:3]
            state.velocity = orbit_states[i][3:]

        states.append(copy.deepcopy(state))

    # Extract times and attitude states
    times = [state.t for state in states]
    attitude_states = np.array([np.concatenate([state.q, state.w]) for state in states])

    return times, attitude_states


def run():
    params = Parameters()
    state = SatelliteState()

    # Define initial state
    state.q = np.array([1, 0, 0, 0])
    state.w = np.array([0.0, 0.2, 0.1])  # [rad/s]

    state.q_d = np.array([0.5, 0.5, 0.5, 0.5])
    state.w_d = np.array([0, 0, 0])  # [rad/s]

    # Define parameters
    params.dt = 1/60  # [sec]
    params.t_end = 2 * 60  # [sec] 2 minutes

    # Simulate
    print("Simulating...")
    states: list[SatelliteState] = []
    num_points = int(params.t_end // params.dt)

    for _ in tqdm(range(num_points)):
        state.propagate_time(params)
        state.propagate_attitude_control(params)

        states.append(copy.deepcopy(state))

    # Plot
    fig = plot_states_plotly(
        states,
        lambda state: state.t,
        {
            "wx": lambda state: state.w[0],
            "wy": lambda state: state.w[1],
            "wz": lambda state: state.w[2],

            "q0": lambda state: state.q[0],
            "q1": lambda state: state.q[1],
            "q2": lambda state: state.q[2],
            "q3": lambda state: state.q[3],

            "Torque x": lambda state: state.control_torque[0],
            "Torque y": lambda state: state.control_torque[1],
            "Torque z": lambda state: state.control_torque[2],

            "Error angle (deg)": lambda state: state.error_angle * 180 / np.pi,
        },
        spacing=0.05
    )

    fig.show()