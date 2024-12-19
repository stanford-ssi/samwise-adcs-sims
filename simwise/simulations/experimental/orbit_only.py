import numpy as np

from simwise.data_structures.parameters import Parameters
from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.simulations.base import run_orbit, run_dispersions
from simwise.utils.plots import plot_results, plot_3D
from simwise.math.frame_transforms import generate_ecef_pn_table

def run():

    # Example Usage
    overrides = {
        # "a": ScalarParameter(7000e3, mean=7000e3, variance=2e3),
        # "e": ScalarParameter(0.005, mean=0.005, variance=0.004),
        "num_dispersions": 1,
        "t_end": 10 * 60 * 60 * 24,
    }
    params = Parameters(**overrides)
    params.ecef_pn_table = generate_ecef_pn_table(params.epoch_jd, params.t_end)
    
    states, times = run_dispersions(
        params=params, 
        runner=run_orbit
    )  
    
    eci_history = np.array([state.r_eci for state in states[0]])
    plot_3D(eci_history, "SAMWISE Orbit Simulation (10 days)")
    # plot_results(states)
    # plot_perturbation_forces(times, drag_forces)