"""
This file contains plotting functions for history objects!

@ Author: Lundeen Cahilly
@ Date: 02-22-2026
"""
import os
import base64
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simwise.constants import R_EARTH, MU_EARTH
from simwise.utils.orbital_elements import state2coe

_EARTH_IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'earth.jpeg')

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


def _load_earth_array():
    from PIL import Image
    return np.array(Image.open(_EARTH_IMAGE_PATH).resize((720, 360)))


def _earth_sphere_trace(img_array):
    """go.Surface of the Earth sphere with texture derived from the image."""
    gray = (0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]) / 255.0
    # Roll so longitude 0° (Prime Meridian) aligns with the ECEF x-axis
    shift = img_array.shape[1] // 2
    gray = np.roll(gray, shift, axis=1)
    img_r = np.roll(img_array, shift, axis=1)

    # Build colorscale: sample actual image RGB at each luminance level
    flat_gray = gray.flatten()
    flat_r = img_r[:,:,0].flatten()
    flat_g = img_r[:,:,1].flatten()
    flat_b = img_r[:,:,2].flatten()
    sort_idx = np.argsort(flat_gray)
    n = 200
    positions = np.linspace(0, len(sort_idx) - 1, n, dtype=int)
    colorscale = []
    for i, pos in enumerate(positions):
        idx = sort_idx[pos]
        colorscale.append([i / (n - 1), f'rgb({flat_r[idx]},{flat_g[idx]},{flat_b[idx]})'])

    N_lat, N_lon = gray.shape
    phi   = np.linspace(0, np.pi, N_lat)
    theta = np.linspace(0, 2 * np.pi, N_lon)
    return go.Surface(
        x=R_EARTH * np.outer(np.sin(phi), np.cos(theta)),
        y=R_EARTH * np.outer(np.sin(phi), np.sin(theta)),
        z=R_EARTH * np.outer(np.cos(phi), np.ones(N_lon)),
        surfacecolor=gray,
        colorscale=colorscale,
        showscale=False, opacity=1.0,
        name="Earth", hoverinfo='skip',
        cmin=0.0, cmax=1.0,
    )


def build_groundtrack_fig(h):
    from simwise.utils.transforms import eci2ecef, ecef2blh
    from simwise.utils.time import gmst as compute_gmst

    lats, lons = [], []
    for s in h:
        r_ecef = eci2ecef(s.r, compute_gmst(s))
        blh = ecef2blh(r_ecef)
        lats.append(np.degrees(blh[0]))
        lons.append(np.degrees(blh[1]))
    
    # wrap longitudes to -180 to 180 degrees
    for i in range(len(lons)):
        lons[i] = (lons[i] + 180) % 360 - 180

    # Break line at antimeridian crossings to avoid horizontal wrap lines
    lons_plot, lats_plot = [lons[0]], [lats[0]]
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > 180:
            lons_plot.append(None)
            lats_plot.append(None)
        lons_plot.append(lons[i])
        lats_plot.append(lats[i])

    with open(_EARTH_IMAGE_PATH, 'rb') as f:
        earth_b64 = base64.b64encode(f.read()).decode()

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=f"data:image/jpeg;base64,{earth_b64}",
        xref="x", yref="y",
        x=-180, y=90,
        sizex=360, sizey=180,
        sizing="stretch",
        opacity=1.0,
        layer="below",
    ))
    fig.add_trace(go.Scatter(
        x=lons_plot, y=lats_plot,
        mode='lines',
        line=dict(color='yellow', width=1),
        name="ground track",
    ))

    fig.add_trace(go.Scatter(
        x=[lons[0]], y=[lats[0]],
        mode='markers',
        marker=dict(size=12, color='lime', symbol='circle',
                    line=dict(color='white', width=2)),
        name="start",
    ))
    fig.add_trace(go.Scatter(
        x=[lons[-1]], y=[lats[-1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='circle',
                    line=dict(color='white', width=2)),
        name="end",
    ))
    fig.update_xaxes(range=[-180, 180], title_text="Longitude [deg]",
                     showgrid=True, gridcolor='rgba(255,255,255,0.3)', dtick=30)
    fig.update_yaxes(range=[-90, 90], title_text="Latitude [deg]",
                     showgrid=True, gridcolor='rgba(255,255,255,0.3)', dtick=30)
    fig.update_layout(title="Ground Track", height=700,
                      plot_bgcolor='black', paper_bgcolor='#111',
                      font=dict(color='white'))
    return fig


def build_ecef_fig(h):
    from simwise.utils.transforms import eci2ecef
    from simwise.utils.time import gmst as compute_gmst

    rx, ry, rz = [], [], []
    for s in h:
        r_ecef = eci2ecef(s.r, compute_gmst(s))
        rx.append(r_ecef[0])
        ry.append(r_ecef[1])
        rz.append(r_ecef[2])

    img_array = _load_earth_array()

    fig = go.Figure()
    fig.add_trace(_earth_sphere_trace(img_array))
    fig.add_trace(go.Scatter3d(
        x=rx, y=ry, z=rz,
        mode='lines',
        line=dict(width=4, color='yellow'),
        name="trajectory",
    ))
    fig.add_trace(go.Scatter3d(
        x=[rx[0]], y=[ry[0]], z=[rz[0]],
        mode='markers',
        marker=dict(size=8, color='lime',
                    line=dict(color='white', width=2)),
        name="start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[rx[-1]], y=[ry[-1]], z=[rz[-1]],
        mode='markers',
        marker=dict(size=8, color='red',
                    line=dict(color='white', width=2)),
        name="end",
    ))
    fig.update_layout(title="ECEF Position", height=700,
                      scene=dict(aspectmode='data'))
    return fig

