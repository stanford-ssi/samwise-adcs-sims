import numpy as np
from simwise.data_structures.parameters import Parameters
import simwise.simulations.simulate_attitude
import simwise.simulations.simulate_orbit

def run():
    sim = IntegratedSimulation()
    orbit_times, orbit_states, attitude_times, attitude_states = sim.run_simulation()
    sim.plot_results(orbit_times, orbit_states, attitude_times, attitude_states)


class IntegratedSimulation:
    def __init__(self):
        self.start_time = Parameters.t_start
        self.end_time = Parameters.t_end
        self.time_step = Parameters.dt
        self.initial_orbit_state = Parameters.initial_orbit_state
        #self.initial_attitude_state = Parameters.initial_attitude_state

    def run_simulation(self):
        
        # Run orbit simulation
        orbit_times, orbit_states = simulate_orbit(
            self.initial_orbit_state,
            self.start_time,
            self.end_time,
            self.time_step
        )

        # Run attitude simulation
        # You may need to adjust this call based on your simulate_attitude function
        attitude_times, attitude_states = simulate_attitude(
            self.start_time,
            self.end_time,
            self.time_step,
            orbit_states  # Passing orbit states to attitude simulation
        )

        return orbit_times, orbit_states, attitude_times, attitude_states

    def store_results(self, time, orbit_state, attitude_state):
        # Implement this method to store or process the simulation results
        # For example, you could append the results to a list or write to a file
        pass

