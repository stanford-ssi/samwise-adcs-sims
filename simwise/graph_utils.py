import plotly.graph_objects as go
from plotly.subplots import make_subplots

def graph_euler(t_arr, y):
    fig = make_subplots(rows=3, cols=1)
    fig.append_trace(go.Scatter(
        x=t_arr, y=y[:, 0],
        name='phi',
        mode='markers',
        marker_color='rgba(152, 0, 0, .1)'
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=t_arr, y=y[:, 1],
        name='theta',
        mode='markers',
        marker_color='rgba(152, 0, 0, .1)'
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=t_arr, y=y[:, 2],
        name='psi',
        mode='markers',
        marker_color='rgba(152, 0, 0, .1)'
    ), row=3, col=1)

    fig.show()

def graph_quaternion(t_arr, y):
    fig = make_subplots(rows=4, cols=1)
    fig.append_trace(go.Scatter(
        x=t_arr, y=y[:, 0],
        name='w',
        mode='markers',
        marker_color='rgba(152, 0, 0, .1)'
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=t_arr, y=y[:, 1],
        name='x',
        mode='markers',
        marker_color='rgba(152, 0, 0, .1)'
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=t_arr, y=y[:, 2],
        name='y',
        mode='markers',
        marker_color='rgba(152, 0, 0, .1)'
    ), row=3, col=1)

    fig.append_trace(go.Scatter(
        x=t_arr, y=y[:, 3],
        name='z',
        mode='markers',
        marker_color='rgba(152, 0, 0, .1)'
    ), row=4, col=1)

    fig.show()

