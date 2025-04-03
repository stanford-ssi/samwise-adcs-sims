import numpy as np

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.utils.plots import plot_results, plot_states_plotly
from simwise.simulations.base import run_dispersions, run_one

def run():

    # Example Usage
    overrides = {
        "w_initial": ArrayParameter([5.0, 4.0, 2.0]),
        "control_mode": "BDOT",

        "Cp": ArrayParameter(np.array([0.1, 0.2, 0.3]), mean=np.array([0.1, 0.2, 0.3]), variance=0.003),
        "Cg": ArrayParameter(np.array([0.2, 0.3, 0.4]), mean=np.array([0.2, 0.3, 0.4]), variance=0.003),
        "num_dispersions": 16,
        "dt_orbit": 120,
        "t_end": 5 * 60,
    }
    params = Parameters(**overrides)
    # states, times = run_one(params)
    # states = np.array([states])
    states, times = run_dispersions(params, runner=run_one)

    plot_results(states)
    
    # # Plot angular velocity
    # fig_mag = plot_states_plotly(
    #     states[0],
    #     lambda state: state.jd,
    #     {
    #         "x": lambda state: state.magnetic_field[0],
    #         "y": lambda state: state.magnetic_field[1],
    #         "z": lambda state: state.magnetic_field[2],
    #     },
    #     spacing=0.05,
    #     title_text="Magnetic Field Vector vs Time",
    # )
    # fig_mag.show()

    # Plot e_angles
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "x": lambda state: state.e_angles[0] / np.linalg.norm(state.e_angles),
            "y": lambda state: state.e_angles[1] / np.linalg.norm(state.e_angles),
            "z": lambda state: state.e_angles[2] / np.linalg.norm(state.e_angles),
        },
        spacing=0.05,
        title_text="Euler Angles vs Time",
    )
    fig_mag.show()

    # Plot q_d
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "q_d0": lambda state: state.q_d[0] / np.linalg.norm(state.q_d),
            "q_d1": lambda state: state.q_d[1] / np.linalg.norm(state.q_d),
            "q_d2": lambda state: state.q_d[2] / np.linalg.norm(state.q_d),
            "q_d3": lambda state: state.q_d[3] / np.linalg.norm(state.q_d),
        },
        spacing=0.05,
        title_text="q vs q_d vs time",
    )
    fig_mag.show()
    
