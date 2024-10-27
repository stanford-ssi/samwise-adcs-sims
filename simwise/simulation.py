import numpy as np
from scipy.integrate import solve_ivp
from simwise.quaternion import quaternion2euler, quaternion_dynamics
from simwise.graph_utils import graph_euler, graph_quaternion

if __name__ == "__main__":
    q = np.array([1, 0, 0, 0])
    Ω = np.array([0.01, 0, 0])
    x0 = np.concatenate((q, Ω))
    inertia = np.array([100, 100, 300])
    tau = np.array([0, 0, 0])
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
        f = lambda t, x: quaternion_dynamics(x, dt, inertia, tau)
        sol = solve_ivp(f, [t, t+dt], x, method='RK45')
        y[i] = sol.y[:, -1]
        x = y[i]
        e_angles[i] = quaternion2euler(y[i, :4])

    graph_euler(t_arr, e_angles)