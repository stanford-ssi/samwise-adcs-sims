import numpy as np
from tqdm import tqdm
import copy

from simwise.data_structures.parameters import Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.utils.plots import plot_states_plotly

def simulate_attitude():
    params = Parameters()
    state = SatelliteState()

    # Set initial state
    state.q = np.array([1, 0, 0, 0])  # Initial quaternion
    state.w = np.array([0.0, 0.2, 0.1])  # Initial angular velocity [rad/s]
    state.q_d = np.array([0.5, 0.5, 0.5, 0.5])  # Desired quaternion
    state.w_d = np.array([0, 0, 0])  # Desired angular velocity [rad/s]

    # Simulate
    print("Simulating attitude...")
    states = []
    num_points = int(params.t_end // params.dt_attitude)

    for _ in tqdm(range(num_points)):
        state.propagate_time(params, params.dt_attitude)
        state.propagate_attitude_control(params)

        states.append(copy.deepcopy(state))

    return states

def run():
    states = simulate_attitude()
    
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

if __name__ == "__main__":
    run()