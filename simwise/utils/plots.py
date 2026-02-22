"""
This file contains plotting functions for history objects!

@ Author: Lundeen Cahilly
@ Date: 02-22-2026
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from simwise.constants import R_EARTH, MU_EARTH
from simwise.utils.orbital_elements import state2coe

def build_attitude_fig(h):
    t = [s.t for s in h]
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Quaternion Components", "Angular Velocity [rad/s]"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    for name, attr in [("q_x", "x"), ("q_y", "y"), ("q_z", "z"), ("q_w", "w")]:
        fig.add_trace(go.Scatter(x=t, y=[getattr(s.q, attr) for s in h], name=name, mode='lines'), row=1, col=1)

    for name, i in [("ω_x", 0), ("ω_y", 1), ("ω_z", 2)]:
        fig.add_trace(go.Scatter(x=t, y=[s.w[i] for s in h], name=name, mode='lines'), row=2, col=1)

    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="[-]", row=1, col=1)
    fig.update_yaxes(title_text="[rad/s]", row=2, col=1)
    fig.update_layout(title="Attitude", height=700)
    return fig

def build_orbit_fig(h):
    t = [s.t for s in h]
    rx = [s.r[0] for s in h]
    ry = [s.r[1] for s in h]
    rz = [s.r[2] for s in h]

    coes = np.array([state2coe(s) for s in h])
    a, e, i, W, w, nu = (
        coes[:, 0],
        coes[:, 1],
        coes[:, 2],
        coes[:, 3],
        coes[:, 4],
        coes[:, 5],
    )

    n_coe = 6
    fig = make_subplots(
        rows=n_coe, cols=2,
        specs=[[{"type": "scatter3d", "rowspan": n_coe}, {"type": "scatter"}]]
             + [[None, {"type": "scatter"}]] * (n_coe - 1),
        column_widths=[0.5, 0.5],
        vertical_spacing=0.04,
    )

    # 3-D trajectory
    fig.add_trace(go.Scatter3d(
        x=rx, y=ry, z=rz,
        mode='lines',
        line=dict(width=4, color='royalblue'),
        name="trajectory",
        showlegend=False,
    ), row=1, col=1)

    # Earth sphere
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    fig.add_trace(go.Surface(
        x=R_EARTH * np.outer(np.cos(u), np.sin(v)),
        y=R_EARTH * np.outer(np.sin(u), np.sin(v)),
        z=R_EARTH * np.outer(np.ones_like(u), np.cos(v)),
        colorscale=[[0, '#1a6b3a'], [1, '#2d9e5c']],
        showscale=False, opacity=0.4, name="Earth", hoverinfo='skip',
    ), row=1, col=1)

    # COE plots
    coe_traces = [
        ("a [m]",   a,  None),
        ("e [-]",   e,  None),
        ("i [deg]", i,  [0, 360]),
        ("Ω [deg]", W,  [0, 360]),
        ("ω [deg]", w,  [0, 360]),
        ("ν [deg]", nu, [0, 360]),
    ]
    for row, (label, vals, ylim) in enumerate(coe_traces, start=1):
        fig.add_trace(go.Scatter(x=t, y=vals, name=label, mode='lines', showlegend=False), row=row, col=2)
        fig.update_yaxes(title_text=label, row=row, col=2, title_standoff=2, range=ylim)
        if row < n_coe:
            fig.update_xaxes(showticklabels=False, row=row, col=2)

    fig.update_xaxes(title_text="Time [s]", row=n_coe, col=2)
    fig.update_layout(title="Orbit", height=1000)
    return fig

