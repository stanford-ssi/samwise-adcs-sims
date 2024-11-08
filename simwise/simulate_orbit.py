import numpy as np
from scipy.integrate import solve_ivp
from quaternion import quaternion2euler, quaternion_dynamics, compute_control_torque, angle_axis_between
from graph_utils import graph_euler, graph_vector_matplotlib, graph_quaternion, graph_quaternion_matplotlib, graph_orbital_elements

from simwise.equinoctial import coe2mee, mee2coe, mee_dynamics
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":

    # Initial orbit conditions
    a = 7000e3 # [m]
    e = 0.001
    i = 0.1 # [rad]
    Ω = 0.1 # [rad]
    ω = 0.1 # [rad]
    θ = 0.1 # [rad]
    mu_earth = 3.986004418e14 # [m^3/s^2]
    orbit = coe2mee(np.array([a, e, i, Ω, ω, θ]))

    dt = 60 # [sec]
    t_end = 60 * 90 # [sec] 2 minutes
    epoch = 0
    num_points = int(t_end // dt)
    t_arr = np.linspace(epoch, t_end, num_points)

    orbital_elem_history = np.zeros((num_points, 6))

    print("Simulating...")
    for i in tqdm(range(num_points)):
        t = t_arr[i]

        f = np.array([0, 0, 0])
        f_orbit = lambda t, orbit: mee_dynamics(orbit, mu_earth, dt, f)
        print("starting to propagate orbit")
        sol_orbit = solve_ivp(f_orbit, [t, t+dt], orbit, method='RK45')
        print("got orbit solution")
        

        orbit = sol_orbit.y[:, -1]
        orbit_keplerian = mee2coe(orbit)
        orbital_elem_history[i] = orbit

    graph_orbital_elements(t_arr, orbital_elem_history)