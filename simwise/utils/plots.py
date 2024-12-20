import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np

from simwise.data_structures.satellite_state import SatelliteState
from simwise import constants


def random_color() -> str:
    """Returns a random rgb color"""
    r = random.randint(0, 200)  # Cap at 200 so not too bright
    g = random.randint(0, 200)
    b = random.randint(0, 200)

    return "#" + bytes([r, g, b]).hex()


def plot_states_plotly(states: list[SatelliteState],
                       x_getter,
                       y_getters,
                       spacing=0.1,
                       x_label="Time [s]",
                       y_label="Value",
                       title_text=""):
    """
    Helper function to plot arbitrary quantities from a list of states.

    Arguments:
        states      List of states
        x_getter    Getter function to get the xaxis (usually time)
        y_getters   Dict mapping titles to y getter functions
    """

    num_plots = len(y_getters)
    x_values = [x_getter(s) for s in states]
    titles = list(y_getters.keys())

    fig = make_subplots(rows=num_plots,
                        cols=1,
                        subplot_titles=titles,
                        shared_xaxes=False,
                        vertical_spacing=spacing)

    colors = [random_color() for i in range(num_plots)]

    for i, series in enumerate(titles):
        y_getter = y_getters[series]
        y_values = [y_getter(s) for s in states]

        fig.add_trace(
            go.Scatter(x=x_values, y=y_values, line=dict(color=colors[i])),
            row=i+1, col=1
        )
        fig.update_yaxes(title_text=y_label, row=i+1, col=1, gridcolor="gray",
            showgrid=True)
        fig.update_xaxes(title_text=x_label, row=i+1, col=1, gridcolor="gray",
            showgrid=True)

    fig.update_layout(
        title=title_text, 
        showlegend=False, 
        height=500 * num_plots,
        plot_bgcolor="white",  # Background of the plotting area
        paper_bgcolor="white"  # Background outside the plotting area
    )

    return fig

def plot_subplots(X, Y, y_axis_titles, x_axis_title, plot_title, save=False):
    """
    Plots N subplots with shared X-axis using Plotly graph objects.

    Parameters:
    Y (numpy.ndarray): A 2D array of shape (N, T) containing the Y data for each subplot.
    X (numpy.ndarray): A 1D array of length T containing the shared X-axis data.
    titles (list): A list of titles for each Y-axis (one for each subplot).

    Returns:
    plotly.graph_objects.Figure: The resulting Plotly figure with subplots.
    """
    
    if len(Y.shape) != 2:
        Y = Y[..., np.newaxis]
    T, N = Y.shape

    # Create subplots with shared X-axis
    fig = make_subplots(
        rows=N,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # Add traces for each subplot
    for i in range(N):
        fig.add_trace(
            go.Scatter(
                x=X,
                y=Y[:, i],
                mode='lines',
                name=y_axis_titles[i]
            ),
            row=i + 1,
            col=1
        )
        # Set Y-axis title for each subplot
        fig.update_yaxes(title_text=y_axis_titles[i], row=i + 1, col=1, gridcolor="gray",
            showgrid=True)
        
        fig.update_xaxes(row=i + 1, col=1, gridcolor="gray",
            showgrid=True)

    # Update layout
    fig.update_layout(
        height=300 * N,  # Adjust the height accordingly
        showlegend=True,
        title=plot_title,
        plot_bgcolor="white",  # Background of the plotting area
        paper_bgcolor="white"  # Background outside the plotting area
    )
    # Set X-axis title only on the bottom subplot
    fig.update_xaxes(title_text=x_axis_title, row=N, col=1, gridcolor="gray",
            showgrid=True)

    # Display the figure
    if save:
        fig.write_image(plot_title)
    else:
        fig.show()

    return fig


def plot_3D(position_history, title=None, plot_earth=True):
    # Separate the components into x, y, and z
    x = position_history[:, 0]
    y = position_history[:, 1]
    z = position_history[:, 2]

    # Create a 3D scatter plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+lines',  # Combines scatter points and lines
        marker=dict(
            size=0.001,
            color='red',  # Optional: Color points based on the z value
            opacity=0.8
        ),
        line=dict(
            color='red',
            width=3
        )
    ))

    # Create a sphere to represent Earth
    if plot_earth:
        earth_radius = constants.EARTH_RADIUS_M
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
        sphere_x = earth_radius * np.cos(u) * np.sin(v)
        sphere_y = earth_radius * np.sin(u) * np.sin(v)
        sphere_z = earth_radius * np.cos(v)

        # Add the sphere to the plot
        fig.add_trace(go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            opacity=0.5,
            showscale=False
        ))

    # Add labels and a title
    if title is None:
        title = "3D Visualization of Output"
    fig.update_layout(
        scene=dict(
        xaxis=dict(
            backgroundcolor="white",
            title="X (m)",  # Set the x-axis title
            gridcolor="gray",  # Set grid color to gray
            showgrid=True      # Ensure the grid is visible
        ),
        yaxis=dict(
            backgroundcolor="white",
            title="Y (m)",  # Set the y-axis title
            gridcolor="gray",
            showgrid=True
        ),
        zaxis=dict(
            backgroundcolor="white",
            title="Z (m)",  # Set the z-axis title
            gridcolor="gray",
            showgrid=True
        )
    ),
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),  # Tight layout
        plot_bgcolor="white",  # Background of the plotting area
        paper_bgcolor="white"  # Background outside the plotting area
    )

    # Show the plot
    fig.show()

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
        random_color = f"rgb({random.randint(0, 200)}, {random.randint(0, 200)}, {random.randint(0, 200)})"

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
        orbit_buffer_list = [0.1, 0.00001, 0.0001, 0.0001, 0.0001, 0.0001]

        for i, (name, func) in enumerate(orbit_params):
            values = [func(state) for state in states]

            # Calculate y-axis range
            y_min = min(values)
            y_max = max(values)
            buffer = orbit_buffer_list[i % len(orbit_buffer_list)] if orbit_buffer_list else 0
            y_range = [y_min - buffer, y_max + buffer]

            fig.add_trace(
                go.Scatter(
                    x=times, y=values, name=name,
                    line=dict(color=random_color),
                    legendgroup=f"run_{run_index}",  # Group traces under the same legend entry
                    showlegend=False  # Only the dummy trace shows in the legend
                ),
                row=i+1, col=1
            )
            fig.update_yaxes(title_text=name, row=i+1, col=1, range=y_range)

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
        
        attitude_params = [
            ("q0_d", lambda s: s.q_d[0]),
            ("q1_d", lambda s: s.q_d[1]),
            ("q2_d", lambda s: s.q_d[2]),
            ("q3_d", lambda s: s.q_d[3]),
            ("wx_d (rad/s)", lambda s: s.w_d[0]),
            ("wy_d (rad/s)", lambda s: s.w_d[1]),
            ("wz_d (rad/s)", lambda s: s.w_d[2])
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
        margin=dict(r=200),  # Adjust right margin to accommodate the legend
        plot_bgcolor="white",  # Background of the plotting area
        paper_bgcolor="white"  # Background outside the plotting area
    )
    fig.update_xaxes(gridcolor="lightgray")
    fig.update_yaxes()

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
        fig.update_xaxes(title_text="Time (s)", row=i+1, col=1, showticklabels=True, gridcolor="lightgray")
        fig.update_xaxes(title_text="Time (s)", row=i+1, col=2, showticklabels=True, gridcolor="lightgray")
        fig.update_yaxes(gridcolor="lightgray", row=i+1, col=1)
        fig.update_yaxes(gridcolor="lightgray", row=i+1, col=2)

    # Adjust subplot titles
    for i, ann in enumerate(fig['layout']['annotations']):
        ann['font'] = dict(size=12)
        if i >= 14:  # Adjusting the position of column titles
            ann['y'] = 1.06

    fig.show()
