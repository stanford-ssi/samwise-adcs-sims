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

def plot3D(position_history):

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

def plot_position_over_jd(positions, julian_dates, save_as=None):
    """
    Plot X, Y, Z positions over Julian Dates in subplots.

    Parameters:
        julian_dates (list or np.array): Array of Julian Dates.
        positions (list or np.array): Nx3 array of positions [X, Y, Z].
        save_as (str, optional): File path to save the plot as an image (e.g., 'position_plot.png').
    """
    # Extract X, Y, Z components
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    z_positions = positions[:, 2]
    
    # Create subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=("X Position", "Y Position", "Z Position"))
    
    # Add X positions
    fig.add_trace(go.Scatter(
        x=julian_dates,
        y=x_positions,
        mode='lines+markers',
        name="X Position"
    ), row=1, col=1)

    # Add Y positions
    fig.add_trace(go.Scatter(
        x=julian_dates,
        y=y_positions,
        mode='lines+markers',
        name="Y Position"
    ), row=2, col=1)

    # Add Z positions
    fig.add_trace(go.Scatter(
        x=julian_dates,
        y=z_positions,
        mode='lines+markers',
        name="Z Position"
    ), row=3, col=1)

    # Update layout
    fig.update_layout(
        title="Position Over Julian Dates",
        xaxis_title="Julian Date",
        yaxis_title="Position",
        height=800,  # Adjust height for better visibility
        showlegend=False  # Legend is unnecessary since each subplot is labeled
    )
    
    # Show the plot
    fig.show()
    
    # Save the plot as an image if requested
    if save_as:
        fig.write_image(save_as)
        print(f"Plot saved as {save_as}")

def plotJD(julian_dates, save_as=None):
    """
    Plot Julian Dates using the array index as the x-axis.

    Parameters:
        julian_dates (list or np.array): Array of Julian Dates to plot.
        save_as (str, optional): File path to save the plot as an image (e.g., 'jd_plot.png').
    """
    # Use the index of the array as the x-axis
    indices = np.array(range(len(julian_dates)))
    
    # Create the line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices,
        y=julian_dates,
        mode='lines+markers',
        name="Julian Date",
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    # Add labels and title
    fig.update_layout(
        title="Julian Dates Over Index",
        xaxis_title="Index",
        yaxis_title="Julian Date",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Show the plot
    fig.show()
    
    # Save the plot as an image if requested
    if save_as:
        fig.write_image(save_as)
        print(f"Plot saved as {save_as}")


def plotJD(julian_dates, save_as=None):
    """
    Plot Julian Dates using the array index as the x-axis.

    Parameters:
        julian_dates (list or np.array): Array of Julian Dates to plot.
        save_as (str, optional): File path to save the plot as an image (e.g., 'jd_plot.png').
    """
    # Use the index of the array as the x-axis
    indices = np.array(range(len(julian_dates)))
    
    # Create the line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices,
        y=julian_dates,
        mode='lines+markers',
        name="Julian Date",
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    # Add labels and title
    fig.update_layout(
        title="Julian Dates Over Index",
        xaxis_title="Index",
        yaxis_title="Julian Date",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Show the plot
    fig.show()
    
    # Save the plot as an image if requested
    if save_as:
        fig.write_image(save_as)
        print(f"Plot saved as {save_as}")

def plot_single(y, julian_dates, save_as=None):
    """
    Plot Julian Dates using the array index as the x-axis.

    Parameters:
        julian_dates (list or np.array): Array of Julian Dates to plot.
        save_as (str, optional): File path to save the plot as an image (e.g., 'jd_plot.png').
    """
    # Use the index of the array as the x-axis
    indices = np.array(range(len(julian_dates)))
    
    # Create the line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=julian_dates,
        y=y,
        mode='lines+markers',
        name="Error [°]",
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    # Add labels and title
    fig.update_layout(
        title="",
        xaxis_title="Index",
        yaxis_title="Error between Approximation and JPL Horizons [°]",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Show the plot
    fig.show()
    
    # Save the plot as an image if requested
    if save_as:
        fig.write_image(save_as)
        print(f"Plot saved as {save_as}")