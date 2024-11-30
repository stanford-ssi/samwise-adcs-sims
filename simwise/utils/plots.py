import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np

from simwise.data_structures.satellite_state import SatelliteState


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
        fig.update_yaxes(title_text=y_label, row=i+1, col=1)
        fig.update_xaxes(title_text=x_label, row=i+1, col=1)

    fig.update_layout(title=title_text, showlegend=False, height=500 * num_plots)

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
        fig.update_yaxes(title_text=y_axis_titles[i], row=i + 1, col=1)

    # Update layout
    fig.update_layout(
        height=300 * N,  # Adjust the height accordingly
        showlegend=True,
        title=plot_title
    )
    # Set X-axis title only on the bottom subplot
    fig.update_xaxes(title_text=x_axis_title, row=N, col=1)

    # Display the figure
    if save:
        fig.write_image(plot_title)
    else:
        fig.show()

    return fig


def plot_3D(position_history):

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
            size=5,
            color=z,  # Optional: Color points based on the z value
            colorscale='Viridis',  # Color scale
            opacity=0.8
        ),
        line=dict(
            color='blue',
            width=2
        )
    ))
    
    # Add labels and a title
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="3D Visualization of Output",
        margin=dict(l=0, r=0, b=0, t=40)  # Tight layout
    )
    
    # Show the plot
    fig.show()