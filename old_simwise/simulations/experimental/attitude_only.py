import numpy as np

from simwise.data_structures.parameters import Parameters
from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.simulations.base import run_attitude, run_dispersions
from simwise.utils.plots import plot_results

def run():

    # Example Usage
    overrides = {
        # for quaternions, set mean to 0 (because we are rotating the quaternion by random Euler angles)
        "q_initial": QuaternionParameter(np.array([1, 0, 0, 0]), variance=np.array([0.1, 0.1, 0.1])),
    }
    params = Parameters(**overrides)
    states, times = run_dispersions(
        params=params, 
        runner=run_attitude
    )

    plot_results(states)
    # plot_perturbation_forces(times, drag_forces)