import numpy as np

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.utils.results import plot_all_results
from simwise.simulations.base import run_dispersions, run_one

def run():

    # Example Usage
    overrides = {
        # for quaternions, set mean to 0 (because we are rotating the quaternion by random Euler angles)
        "q_initial": QuaternionParameter(np.array([0.65194207, 0.7509002, -0.03757633, 0.09853141]), variance=np.array([0.1, 0.1, 0.1])),
        # "pointing_mode": "NadirPointingVelocityConstrained",
        "num_dispersions": 16,
        "attitude_determination_mode": "TRIAD",
        # "pointing_mode": "SunPointingNadirConstrained",
        "dt_orbit": 120,
        "t_end": 45 * 60,
    }
    params = Parameters(**overrides)
    # states, times = run_one(params)
    # states = np.array([states])
    states, times = run_dispersions(params, runner=run_one)

    plot_all_results(states)
