import numpy as np
from tqdm import tqdm
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count
import random

from simwise.data_structures.parameters import ArrayParameter, QuaternionParameter, ScalarParameter, Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.orbit.equinoctial import coe2mee, mee2coe

def run_simulation(params):
    params = params
    state = SatelliteState()

    # Initial orbit conditions
    state.orbit_keplerian = np.array([
        params.a, params.e, params.i,
        params.Ω, params.ω, params.θ
    ])
    state.orbit_mee = coe2mee(state.orbit_keplerian)

    # Initial attitude conditions
    state.q = params.q_initial
    state.w = params.w_initial
    state.q_d = params.q_desired
    state.w_d = params.w_desired

    states = []
    times = []
    num_points_attitude = int((params.t_end - params.t_start) // params.dt_attitude) + 1
    num_points_orbit = int((params.t_end - params.t_start) // params.dt_orbit) + 1

    for i in range(num_points_attitude):
        # Define time in terms of smaller timestep - attitude
        t = params.t_start + i * params.dt_attitude
        state.t = t  # Set the time attribute of the state

        # Propagate attitude at every step - smaller timestep
        state.propagate_attitude_control(params)
        
        # Propagate orbit for greater time step - orbit
        if i % int(params.dt_orbit / params.dt_attitude) == 0:
            state.propagate_orbit(params)
        
        # Calculate perturbation forces
        state.calculate_pertubation_forces(params)
        
        states.append(copy.deepcopy(state))
        times.append(t)

    return states, times
    
def main(dispersed_instances, num_dispersions):
    num_workers = cpu_count()
    states_from_dispersions = []
    times_from_dispersions = []
    with Pool(processes=num_workers) as pool:
        with tqdm(total=num_dispersions, desc=f"Simulating {num_dispersions} runs") as pbar:
            for states, times in pool.imap_unordered(run_simulation, dispersed_instances):
                states_from_dispersions.append(states)
                times_from_dispersions.append(times)
                pbar.update()
    return states_from_dispersions, times_from_dispersions


def plot_results(states_from_dispersions):
    # Create subplots
    fig = make_subplots(
        rows=7, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.2,  # Increase spacing between columns
        subplot_titles=(
            "Semi-major axis", "q_0", 
            "Eccentricity", "q_1",
            "Inclination", "q_2",
            "RAAN", "q_3",
            "Argument of Periapsis", "ω_x",
            "True Anomaly", "ω_y",
            "", "ω_z"
        ),
    )

    for run_index, states in enumerate(states_from_dispersions, start=1):
        # Generate a random color for this set of states
        random_color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

        # Add a dummy trace to the legend for this run
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],  # Dummy trace
                mode="markers",
                marker=dict(color=random_color, size=10),
                name=f"Run {run_index}",
                legendgroup=f"run_{run_index}",  # Group traces under the same legend entry
                showlegend=True  # Show this trace in the legend
            )
        )

        times = [state.t for state in states]  # Extract times from states

        # Orbit parameters
        orbit_params = [
            ("a (km)", lambda s: s.orbit_keplerian[0] / 1000),
            ("e", lambda s: s.orbit_keplerian[1]),
            ("i (deg)", lambda s: np.degrees(s.orbit_keplerian[2])),
            ("Ω (deg)", lambda s: np.degrees(s.orbit_keplerian[3])),
            ("ω (deg)", lambda s: np.degrees(s.orbit_keplerian[4])),
            ("θ (deg)", lambda s: np.degrees(s.orbit_keplerian[5]))
        ]

        for i, (name, func) in enumerate(orbit_params):
            values = [func(state) for state in states]
            fig.add_trace(
                go.Scatter(
                    x=times, y=values, name=name,
                    line=dict(color=random_color),
                    legendgroup=f"run_{run_index}",  # Group traces under the same legend entry
                    showlegend=False  # Only the dummy trace shows in the legend
                ),
                row=i+1, col=1
            )
            fig.update_yaxes(title_text=name, row=i+1, col=1)

        # Attitude parameters
        attitude_params = [
            ("q0", lambda s: s.q[0]),
            ("q1", lambda s: s.q[1]),
            ("q2", lambda s: s.q[2]),
            ("q3", lambda s: s.q[3]),
            ("wx (rad/s)", lambda s: s.w[0]),
            ("wy (rad/s)", lambda s: s.w[1]),
            ("wz (rad/s)", lambda s: s.w[2])
        ]

        for i, (name, func) in enumerate(attitude_params):
            values = [func(state) for state in states]
            fig.add_trace(
                go.Scatter(
                    x=times, y=values, name=name,
                    line=dict(color=random_color),
                    legendgroup=f"run_{run_index}",  # Group traces under the same legend entry
                    showlegend=False  # Only the dummy trace shows in the legend
                ),
                row=i+1, col=2
            )
            fig.update_yaxes(title_text=name, row=i+1, col=2)

    # Update layout
    fig.update_layout(
        height=None,  # Automatically adjusts to the content
        width=None,   # Automatically adjusts to the content
        autosize=True,
        title_text="Integrated Orbit and Attitude Simulation",
        title_x=0.5,
        title_y=0.99,
        title_xanchor='center',
        title_yanchor='top',
        legend=dict(
            x=1.1,  # Position the legend just outside the plotting area
            y=1.0,
            xanchor="left",
            yanchor="top",
            title="Run Legend",
            font=dict(size=12),
            bordercolor="black",
            borderwidth=1
        ),
        margin=dict(r=200)  # Adjust right margin to accommodate the legend
    )

    # Add column titles
    fig.add_annotation(
        x=0.25, y=1.05, xref="paper", yref="paper",
        text="Orbit Parameters", showarrow=False, font=dict(size=16)
    )
    fig.add_annotation(
        x=0.75, y=1.05, xref="paper", yref="paper",
        text="Attitude Parameters", showarrow=False, font=dict(size=16)
    )

    # Update x-axes to show time
    for i in range(7):
        fig.update_xaxes(title_text="Time (s)", row=i+1, col=1, showticklabels=True)
        fig.update_xaxes(title_text="Time (s)", row=i+1, col=2, showticklabels=True)

    # Adjust subplot titles
    for i, ann in enumerate(fig['layout']['annotations']):
        ann['font'] = dict(size=12)
        if i >= 14:  # Adjusting the position of column titles
            ann['y'] = 1.06

    fig.show()


def run():

    # Example Usage
    overrides = {
        # for quaternions, set mean to 0 (because we are rotating the quaternion by random Euler angles)
        "q_initial": QuaternionParameter(np.array([1, 0, 0, 0]), variance=np.array([0.1, 0.1, 0.1])),

        "Cp": ArrayParameter(np.array([0.1, 0.2, 0.3]), mean=np.array([0.1, 0.2, 0.3]), variance=0.003),
        "Cg": ArrayParameter(np.array([0.2, 0.3, 0.4]), mean=np.array([0.2, 0.3, 0.4]), variance=0.003),
    }

    params = Parameters(**overrides)

    # Generate 3 dispersed instances
    num_dispersions = 20
    dispersed_instances = params.generate_dispersions(num_dispersions)

    # print all values that are dispersed
    for dispersed_params in dispersed_instances:
        for attr in dir(dispersed_params):
            if not attr.startswith("_") and isinstance(getattr(dispersed_params, attr), (ArrayParameter, QuaternionParameter, ScalarParameter)):
                # print(attr, getattr(dispersed_params, attr))
                pass
    states, times = main(dispersed_instances=dispersed_instances, num_dispersions=num_dispersions)
    plot_results(states)
    # sim.plot_perturbation_forces(times, drag_forces)

if __name__ == "__main__":
    run()