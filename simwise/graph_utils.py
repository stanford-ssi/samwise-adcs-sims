import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

def graph_vector_matplotlib(t_arr, y, name1="x", name2="y", name3="z"):
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    axs[0].plot(t_arr, y[:, 0], color='b', label=name1)
    axs[0].set_xlabel('t')
    axs[0].set_ylabel(name1)
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t_arr, y[:, 1], color='b', label=name2)
    axs[1].set_xlabel('t')
    axs[1].set_ylabel(name2)
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t_arr, y[:, 2], color='b', label=name3)
    axs[2].set_xlabel('t')
    axs[2].set_ylabel(name3)
    axs[2].legend()
    axs[2].grid()

    plt.show()


def graph_quaternion_matplotlib(t_arr, y):
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))

    axs[0].plot(t_arr, y[:, 0], color='b', label='w')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('w')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t_arr, y[:, 1], color='b', label='x')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('x')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t_arr, y[:, 2], color='b', label='y')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('y')
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(t_arr, y[:, 3], color='b', label='z')
    axs[3].set_xlabel('t')
    axs[3].set_ylabel('z')
    axs[3].legend()
    axs[3].grid()
    plt.show()

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

