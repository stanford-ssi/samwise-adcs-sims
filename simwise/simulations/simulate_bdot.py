# Simulate control with bdot
import numpy as np
from tqdm import tqdm
import copy

from simwise.data_structures.parameters import Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.utils.plots import plot_states_plotly
from simwise.orbit.equinoctial import *


def magnetic_field(t):
    # Time-varying magnetic field (simulating orbital motion)
    # This creates a rotating magnetic field as might be seen in orbit (NOT ACCURATE W/ POLE BEHAVIOR)
    B_magnitude = 4.5e-5  # ~45 μT, typical in LEO
    omega_orbit = 2 * np.pi / 5600  # Approximate orbital angular velocity

    Bx = B_magnitude * np.sin(omega_orbit * t)
    By = B_magnitude * np.cos(omega_orbit * t)
    Bz = B_magnitude * 0.5 * np.sin(2 * omega_orbit * t)

    return np.array([Bx, By, Bz])


def run():
    params = Parameters()
    state = SatelliteState()

    # Define initial state
    state.q = np.array([1, 0, 0, 0])
    state.w = np.array([0.1, -0.15, 0.08])  # Initial angular velocity [rad/s]

    # Simulation parameters
    params. dt = 5  # [sec]
    params.t_end = 24 * 60 * 60  # 8 hour simulation

    # Simulate
    print("Simulating...")
    states: list[SatelliteState] = []
    num_points = int(params.t_end // params.dt)

    for _ in tqdm(range(num_points)):
        state.propagate_time(params)

        # Set current magnetic field - TODO: replace with world model
        state.B = magnetic_field(state.t)

        state.propagate_attitude_bdot(params)
        states.append(copy.deepcopy(state))

    # Plot angular velocity
    fig_mag = plot_states_plotly(
        states,
        lambda state: state.t,
        {
            "Angular Velocity Magnitude [rad/s]": lambda state: np.linalg.norm(state.w),
        },
        spacing=0.05,
        title_text="Detumbling Performance",
    )
    fig_mag.show()

    fig_omega = plot_states_plotly(
        states,
        lambda state: state.t,
        {
            "ωx": lambda state: state.w[0],
            "ωy": lambda state: state.w[1],
            "ωz": lambda state: state.w[2],
        },
        spacing=0.05,
        y_label="Angular velocity [rad/s]",
        title_text="Angular Velocities vs Time",
    )
    fig_omega.show()

    # Magnetic moments
    fig_mu = plot_states_plotly(
        states,
        lambda state: state.t,
        {
            "μx": lambda state: state.mu[0],
            "μy": lambda state: state.mu[1],
            "μz": lambda state: state.mu[2],
        },
        spacing=0.05,
        y_label="Magnetic moment [A⋅m²]",
        title_text="Magnetic Moments vs Time",
    )

    fig_mu.show()

    # Magnetic field
    fig_B = plot_states_plotly(
        states,
        lambda state: state.t,
        {
            "Bx": lambda state: state.B[0],
            "By": lambda state: state.B[1],
            "Bz": lambda state: state.B[2],
        },
        spacing=0.05,
        y_label="Magnetic field [T]",
        title_text="Magnetic field [T]",
    )
    fig_B.show()
