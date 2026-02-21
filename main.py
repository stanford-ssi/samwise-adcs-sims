"""
Full attitude propagator using simwise.

@ Author: Lundeen Cahilly
@ Date: 2026-02-20
"""

import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simwise.math import Quaternion, rk4
from simwise.satellite import SatelliteState, SatelliteParams
from simwise.forces import j2_perturbation
from simwise.torques import gravity_gradient
from simwise.dynamics import state_dot, attitude_dot, orbit_dot
from simwise.constants import R_EARTH

I_body = np.array([[0.01861, 0.00529, 0.0001439],
              [0.00529, 0.01833, 0.0000584709],
              [0.0001439, 0.0000584709, 0.01558]])  # [kg m^2]
m = 2.0 # [kg]

def propagate():
    params = SatelliteParams(I=I_body, m=m)
    state = SatelliteState(
        q_eci2body=Quaternion(0, 0, 0, 1),
        w_eci=np.array([0.0, 0.0, 0.1]),
        r_eci=np.array([R_EARTH + 350e3, 0.0, 0.0]),
        v_eci=np.array([0, 5445.48, 5445.48]),
        t=0.0,
    )
    dt = 0.1
    orbit_every = 100
    dt_orbit = dt * orbit_every
    tf = 0.5 * 3600.0 # [s]

    trajectory = []
    n_steps = int(tf / dt)
    f_orbit = lambda s, t: orbit_dot(s, params, perturbations=[])
    f_attitude = lambda s, t: attitude_dot(s, params, torques=[gravity_gradient])
    with tqdm(total=n_steps, desc="Propagating", unit="step") as pbar:
        step = 0
        while state.t < tf:
            trajectory.append(state)
            if step % orbit_every == 0: # propagate orbit a lot less often than attitude
                state_orbit = rk4(state, dt_orbit, f_orbit)
            state = rk4(state, dt, f_attitude)
            state.r = state_orbit.r
            state.v = state_orbit.v
            step += 1
            pbar.update(1)

    return trajectory

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

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d", "colspan": 2}, None],
               [{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=("3-D Orbit (ECI)", "Position Components [m]", "Velocity Components [m/s]"),
        vertical_spacing=0.12,
    )

    # 3-D trajectory
    fig.add_trace(go.Scatter3d(
        x=rx, y=ry, z=rz,
        mode='lines',
        line=dict(width=4, color='royalblue'),
        name="trajectory",
        showlegend=False,
    ), row=1, col=1)

    # Earth sphere for reference
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    ex = R_EARTH * np.outer(np.cos(u), np.sin(v))
    ey = R_EARTH * np.outer(np.sin(u), np.sin(v))
    ez = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        colorscale=[[0, '#1a6b3a'], [1, '#2d9e5c']],
        showscale=False, opacity=0.4, name="Earth",
        hoverinfo='skip',
    ), row=1, col=1)

    # Position components vs time
    for name, vals in [("r_x", rx), ("r_y", ry), ("r_z", rz)]:
        fig.add_trace(go.Scatter(x=t, y=vals, name=name, mode='lines'), row=2, col=1)

    # Velocity components vs time
    for name, i in [("v_x", 0), ("v_y", 1), ("v_z", 2)]:
        fig.add_trace(go.Scatter(x=t, y=[s.v[i] for s in h], name=name, mode='lines'), row=2, col=2)

    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="[m]", row=2, col=1)
    fig.update_yaxes(title_text="[m/s]", row=2, col=2)
    fig.update_layout(title="Orbit", height=900)
    return fig

history = propagate()
build_attitude_fig(history).show()
build_orbit_fig(history).show()
