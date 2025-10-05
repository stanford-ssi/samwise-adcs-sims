import argparse

import numpy as np

from simwise.utils.plots import plot_states_plotly, plot_3D, plot_results
from simwise.math.quaternion import quaternions_to_axis_angle

def plot_all_results(states):
    plot_results(states)

    # Plot error angle vs time
    fig = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "error angle [°]": lambda state: np.rad2deg(quaternions_to_axis_angle(state.x_k[:4], state.q)[0])
        },
        spacing=0.05,
        title_text="Error Angle vs Time",
    )
    fig.show()
    
    # Plot mag meas vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "x": lambda state: state.v_mag_eci[0] / np.linalg.norm(state.v_mag_eci),
            "y": lambda state: state.v_mag_eci[1] / np.linalg.norm(state.v_mag_eci),
            "z": lambda state: state.v_mag_eci[2] / np.linalg.norm(state.v_mag_eci),
        },
        spacing=0.05,
        title_text="Mag Vector ECI Model vs Time",
    )
    fig_mag.show()

    # Plot mag meas vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "x": lambda state: state.v_mag_meas[0] / np.linalg.norm(state.v_mag_meas),
            "y": lambda state: state.v_mag_meas[1] / np.linalg.norm(state.v_mag_meas),
            "z": lambda state: state.v_mag_meas[2] / np.linalg.norm(state.v_mag_meas),
        },
        spacing=0.05,
        title_text="Mag Vector Measurement vs Time",
    )
    fig_mag.show()

    # Plot sun meas vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "x": lambda state: state.v_sun_meas[0] / np.linalg.norm(state.v_sun_meas),
            "y": lambda state: state.v_sun_meas[1] / np.linalg.norm(state.v_sun_meas),
            "z": lambda state: state.v_sun_meas[2] / np.linalg.norm(state.v_sun_meas),
        },
        spacing=0.05,
        title_text="OSPF Sun Vector Measurement vs Time",
    )
    fig_mag.show()

    # Plot nadir vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "q1": lambda state: state.x_k[0],
            "q2": lambda state: state.x_k[1],
            "q3": lambda state: state.x_k[2],
            "q4": lambda state: state.x_k[3],
        },
        spacing=0.05,
        title_text="Filtered Quaternion",
    )
    fig_mag.show()

    # Plot nadir vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "q1": lambda state: state.attitude_knowedge_error[0],
            "q2": lambda state: state.attitude_knowedge_error[1],
            "q3": lambda state: state.attitude_knowedge_error[2],
            "q4": lambda state: state.attitude_knowedge_error[3],
        },
        spacing=0.05,
        title_text="Filtered Quaternion Error",
    )
    fig_mag.show()

    # Plot nadir vector
    ecef_history = np.array([state.r_ecef for state in states[0]])
    plot_3D(ecef_history)

    # Plot nadir vector
    eci_history = np.array([state.r_eci for state in states[0]])
    plot_3D(eci_history)

    # Plot nadir vector
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "lat": lambda state: state.lla_wgs84[0],
            "lon": lambda state: state.lla_wgs84[1],
            "altitude": lambda state: state.lla_wgs84[2],
        },
        spacing=0.05,
        title_text="WGS84 LLA vs Time",
    )
    fig_mag.show()

    # Plot eclipse
    fig_mag = plot_states_plotly(
        states[0],
        lambda state: state.t,
        {
            "eclipse": lambda state: state.eclipse,
        },
        spacing=0.05,
        title_text="Eclipse vs Time",
    )
    fig_mag.show()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("states_file")

    args = parser.parse_args()

    # load states from file
    states = np.load(args.states_file, allow_pickle=True)

    plot_all_results(states)