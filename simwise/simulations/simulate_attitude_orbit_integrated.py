from simwise.data_structures.parameters import Parameters
from simwise.simulations.simulate_attitude import run as simulate_attitude
from simwise.simulations.simulate_orbit import run as simulate_orbit
import matplotlib.pyplot as plt
import numpy as np

def run():
    sim = IntegratedSimulation()
    orbit_times, orbit_states, attitude_times, attitude_states = sim.run_simulation()
    sim.plot_results(orbit_times, orbit_states, attitude_times, attitude_states)

class IntegratedSimulation:
    def __init__(self):
        self.params = Parameters()

    def run_simulation(self):
        # Run orbit simulation
        orbit_states, orbit_times = simulate_orbit()

        # Run attitude simulation
        attitude_states, attitude_times = simulate_attitude()

        return orbit_times, orbit_states, attitude_times, attitude_states

    def plot_results(self, orbit_times, orbit_states, attitude_times, attitude_states):
        # Plot orbit results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        ax1.plot(orbit_states[:, 0], orbit_states[:, 1], label='Orbit XY')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Orbit Trajectory')
        ax1.legend()
        ax1.grid(True)

        # Plot attitude quaternions
        ax2.plot(attitude_times, attitude_states[:, 0], label='q0')
        ax2.plot(attitude_times, attitude_states[:, 1], label='q1')
        ax2.plot(attitude_times, attitude_states[:, 2], label='q2')
        ax2.plot(attitude_times, attitude_states[:, 3], label='q3')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Quaternion Components')
        ax2.set_title('Attitude Quaternions')
        ax2.legend()
        ax2.grid(True)

        # Plot angular velocities
        ax3.plot(attitude_times, attitude_states[:, 4], label='wx')
        ax3.plot(attitude_times, attitude_states[:, 5], label='wy')
        ax3.plot(attitude_times, attitude_states[:, 6], label='wz')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angular Velocity (rad/s)')
        ax3.set_title('Angular Velocities')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()