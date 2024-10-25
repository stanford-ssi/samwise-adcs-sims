import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def normalize_quaternion(q):
    """_summary_

    Args:
        q (_type_): _description_
    """
    return q / np.linalg.norm(q)

def attitude_dynamics(x, dt, inertia, tau):
    """_summary_

    Args:
        x (_type_): _description_
        dt (_type_): _description_
        inertia (_type_): _description_
        tau (_type_): _description_
    """
    Q = normalize_quaternion(x[:4])
    Ω = x[4:]
    quaternion_dynamics = np.array([
        [0,   -Ω[0], -Ω[1], -Ω[2]],
        [Ω[0],    0,  Ω[2], -Ω[1]],
        [Ω[1],-Ω[2],     0,  Ω[0]],
        [Ω[2], Ω[1], -Ω[0],     0]
    ])
    Q_dot = 0.5 * quaternion_dynamics @ Q
    
    Ω_dot = np.array([
        (inertia[1] - inertia[2])/inertia[0] * Ω[1] * Ω[2] + tau[0] / inertia[0],
        (inertia[2] - inertia[0])/inertia[1] * Ω[0] * Ω[2] + tau[1] / inertia[1],
        (inertia[0] - inertia[1])/inertia[2] * Ω[0] * Ω[1] + tau[2] / inertia[2]
    ])
    return np.concatenate((Q_dot, Ω_dot))


def quaternion2euler(q, sequence="zyx"):
    """_summary_

    Args:
        q (_type_): _description_
        sequence (_type_): _description_
    """
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    if sequence == "zyx":
        phi = np.arctan2((2*(q1*q2 + q3*q4)), 1 - 2*(q2**2 + q3**2))
        theta = np.arcsin(2*(q1*q3 - q4*q2))
        psi = np.arctan2(2*(q1*q4 + q2*q3), 1 - 2*(q3**2 + q4**2))
        return np.array([phi, theta, psi])
    else:
        raise ValueError("Invalid sequence")


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


if __name__ == "__main__":
    q = np.array([1, 0, 0, 0])
    Ω = np.array([0.01, 0, 0])
    x0 = np.concatenate((q, Ω))
    inertia = np.array([100, 100, 300])
    tau = np.array([0, 0, 0])
    # x_dot = attitude_dynamics(x, dt, inertia, tau)
    dt = 1/60 # [sec]
    t_end = 10.47 * 60 # [sec] 10 minutes
    epoch = 0
    num_points = int(t_end // dt)
    

    y = np.zeros((num_points, 7))
    t_arr = np.arange(num_points) * dt + epoch
    x = x0
    e_angles = np.zeros((num_points, 3))

    for i in range(num_points):
        t = t_arr[i]
        f = lambda t, x: attitude_dynamics(x, dt, inertia, tau)
        sol = solve_ivp(f, [t, t+dt], x, method='RK45')
        y[i] = sol.y[:, -1]
        x = y[i]
        e_angles[i] = quaternion2euler(y[i, :4])

    graph_euler(t_arr, e_angles)