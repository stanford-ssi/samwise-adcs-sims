import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

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