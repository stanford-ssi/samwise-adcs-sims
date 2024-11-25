import numpy as np
from tqdm import tqdm
import copy

from simwise.data_structures.parameters import Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.utils.plots import plot_states_plotly
from simwise.orbit.equinoctial import coe2mee, mee2coe

def simulate_orbit():
    params = Parameters()
    state = SatelliteState()

    # Use parameters defined in parameters.py
    state.orbit_keplerian = np.array([
        params.a,
        params.e,
        params.i,
        params.Ω,
        params.ω,
        params.θ
    ])
    state.orbit_mee = coe2mee(state.orbit_keplerian)

    # Simulate
    print("Simulating orbit...")
    states = []
    times = []
    num_points = int((params.t_end - params.t_start) // params.dt_orbit) + 1

    for i in tqdm(range(num_points)):
        t = params.t_start + i * params.dt_orbit
        state.propagate_orbit(params)
        states.append(copy.deepcopy(state))
        times.append(t)

    return states, times

def run():
    states, times = simulate_orbit()
    
    # Plot
    fig = plot_states_plotly(
        states,
        lambda state: state.t,
        {
            "a": lambda state: state.orbit_keplerian[0],
            "e": lambda state: state.orbit_keplerian[1],
            "i": lambda state: state.orbit_keplerian[2],
            "Ω": lambda state: state.orbit_keplerian[3],
            "ω": lambda state: state.orbit_keplerian[4],
            "θ": lambda state: state.orbit_keplerian[5],

            "p": lambda state: state.orbit_mee[0],
            "f": lambda state: state.orbit_mee[1],
            "g": lambda state: state.orbit_mee[2],
            "h": lambda state: state.orbit_mee[3],
            "k": lambda state: state.orbit_mee[4],
            "L": lambda state: state.orbit_mee[5],
        },
        spacing=0.05
    )
    fig.show()

if __name__ == "__main__":
    run()