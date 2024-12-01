from simwise.data_structures.parameters import Parameters
from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.simulations.base import run_orbit, run_dispersions
from simwise.utils.plots import plot_results

def run():

    # Example Usage
    overrides = {
        "a": ScalarParameter(7000e3, mean=7000e3, variance=2e3),
        "e": ScalarParameter(0.005, mean=0.005, variance=0.004),
        "num_dispersions": 20
    }
    params = Parameters(**overrides)
    states, times = run_dispersions(
        params=params, 
        runner=run_orbit
    )

    plot_results(states)
    # plot_perturbation_forces(times, drag_forces)