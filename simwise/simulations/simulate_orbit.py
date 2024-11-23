import numpy as np
from tqdm import tqdm
import copy

from simwise.data_structures.parameters import Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.utils.plots import plot_states_plotly
from simwise.orbit.equinoctial import *

def run():
    params = Parameters()
    state = SatelliteState()

    # Set parameters
    params.dt = 1 # [sec]
    params.t_end = 60 * 90 # [sec] 
    
    # Initial orbit conditions
    a = 7000e3 # [m]
    e = 0.001
    i = 0.1 # [rad]
    Ω = 0.1 # [rad]
    ω = 0.1 # [rad]
    θ = 0.1 # [rad]
    state.orbit_keplerian = np.array([a, e, i, Ω, ω, θ])
    state.orbit_mee = coe2mee(state.orbit_keplerian)

    # Simulate
    print("Simulating...")
    states: list[SatelliteState] = []
    num_points = int(params.t_end // params.dt)

    for _ in tqdm(range(num_points)):
        state.propagate_time(params)
        state.propagate_orbit(params)

        states.append(copy.copy(state))

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