import numpy as np

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.utils.results import plot_all_results
from simwise.simulations.base import run_dispersions, run_one

def run():

    # Example Usage
    overrides = {
        # for quaternions, set mean to 0 (because we are rotating the quaternion by random Euler angles)
        "q_initial": QuaternionParameter(np.array([1, 0, 0, 0]), variance=np.array([0.1, 0.1, 0.1])),

        "Cp": ArrayParameter(np.array([0.1, 0.2, 0.3]), mean=np.array([0.1, 0.2, 0.3]), variance=0.003),
        "Cg": ArrayParameter(np.array([0.2, 0.3, 0.4]), mean=np.array([0.2, 0.3, 0.4]), variance=0.003),
        # "num_dispersions": 16,
        # "i": 0.0,
        "e": 0.0,
        # "pointing_mode": "NadirPointingVelocityConstrained",
        "attitude_determination_mode": "TRIAD",
        "dt_orbit": 120,
        "t_end": 45 * 60,
    }
    params = Parameters(**overrides)
    # states, times = run_one(params)
    # states = np.array([states])
    states, times = run_dispersions(params, runner=run_one)

    plot_all_results(states)
