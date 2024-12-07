import numpy as np

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.utils.plots import plot_results, plot_states_plotly
from simwise.simulations.base import run_dispersions, run_one

def run():

    # Example Usage
    overrides = {
        # for quaternions, set mean to 0 (because we are rotating the quaternion by random Euler angles)
        "q_initial": QuaternionParameter(np.array([1, 0, 0, 0]), variance=np.array([0.1, 0.1, 0.1])),

        "Cp": ArrayParameter(np.array([0.1, 0.2, 0.3]), mean=np.array([0.1, 0.2, 0.3]), variance=0.003),
        "Cg": ArrayParameter(np.array([0.2, 0.3, 0.4]), mean=np.array([0.2, 0.3, 0.4]), variance=0.003),
        # "num_dispersions": 16,
        "dt_orbit": 120,
        "t_end": 10 * 60,
    }
    params = Parameters(**overrides)
    states, times = run_one(params)
    states = np.array([states])
    # states, times = run_dispersions(params, runner=run_one)

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

    # Plot sun vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "x": lambda state: state.r_sun_eci[0] / np.linalg.norm(state.r_sun_eci),
            "y": lambda state: state.r_sun_eci[1] / np.linalg.norm(state.r_sun_eci),
            "z": lambda state: state.r_sun_eci[2] / np.linalg.norm(state.r_sun_eci),
        },
        spacing=0.05,
        title_text="Sun Vector vs Time",
    )
    fig_mag.show()

    # Plot nadir vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.jd,
        {
            "x": lambda state: state.r_eci[0] / np.linalg.norm(state.r_eci),
            "y": lambda state: state.r_eci[1] / np.linalg.norm(state.r_eci),
            "z": lambda state: state.r_eci[2] / np.linalg.norm(state.r_eci),
        },
        spacing=0.05,
        title_text="Nadir Vector vs Time",
    )
    fig_mag.show()
    
