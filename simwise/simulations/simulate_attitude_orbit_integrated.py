import numpy as np
from tqdm import tqdm
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simwise.data_structures.parameters import Parameters
from simwise.data_structures.satellite_state import SatelliteState
from simwise.orbit.equinoctial import coe2mee, mee2coe

class IntegratedSimulation:
    def __init__(self):
        self.params = Parameters()
        self.state = SatelliteState()

        # Initial orbit conditions
        self.state.orbit_keplerian = np.array([
            self.params.a, self.params.e, self.params.i,
            self.params.Ω, self.params.ω, self.params.θ
        ])
        self.state.orbit_mee = coe2mee(self.state.orbit_keplerian)

        # Initial attitude conditions
        self.state.q = self.params.q_initial
        self.state.w = self.params.w_initial
        self.state.q_d = self.params.q_desired
        self.state.w_d = self.params.w_desired

    def run_simulation(self):
        print("Simulating...")
        states = []
        times = []
        num_points_attitude = int((self.params.t_end - self.params.t_start) // self.params.dt_attitude) + 1
        num_points_orbit = int((self.params.t_end - self.params.t_start) // self.params.dt_orbit) + 1

        for i in tqdm(range(num_points_attitude)):
            # Define time in terms of smaller timestep - attitude
            t = self.params.t_start + i * self.params.dt_attitude
            self.state.t = t  # Set the time attribute of the state

            # Propagate attitude at every step - smaller timestep
            self.state.propagate_attitude_control(self.params)
            
            # Propagate orbit for greater time step - orbit
            if i % int(self.params.dt_orbit / self.params.dt_attitude) == 0:
                self.state.propagate_orbit(self.params)
            
            states.append(copy.deepcopy(self.state))
            times.append(t)

        return states
    
    def plot_results(self, states):
        # Create subplots
        fig = make_subplots(
            rows=7, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.15,
            subplot_titles=(
                "Semi-major axis", "q_0", 
                "Eccentricity", "q_1",
                "Inclination", "q_2",
                "RAAN", "q_3",
                "Argument of Periapsis", "ω_x",
                "True Anomaly", "ω_y",
                "", "ω_z"
            ),
        )

        times = [state.t for state in states]  # Extract times from states

        # Orbit parameters
        orbit_params = [
            ("a (km)", lambda s: s.orbit_keplerian[0] / 1000),
            ("e", lambda s: s.orbit_keplerian[1]),
            ("i (deg)", lambda s: np.degrees(s.orbit_keplerian[2])),
            ("Ω (deg)", lambda s: np.degrees(s.orbit_keplerian[3])),
            ("ω (deg)", lambda s: np.degrees(s.orbit_keplerian[4])),
            ("θ (deg)", lambda s: np.degrees(s.orbit_keplerian[5]))
        ]

        for i, (name, func) in enumerate(orbit_params):
            values = [func(state) for state in states]
            fig.add_trace(go.Scatter(x=times, y=values, name=name), row=i+1, col=1)
            fig.update_yaxes(title_text=name, row=i+1, col=1)

        # Attitude parameters
        attitude_params = [
            ("q0", lambda s: s.q[0]),
            ("q1", lambda s: s.q[1]),
            ("q2", lambda s: s.q[2]),
            ("q3", lambda s: s.q[3]),
            ("wx (rad/s)", lambda s: s.w[0]),
            ("wy (rad/s)", lambda s: s.w[1]),
            ("wz (rad/s)", lambda s: s.w[2])
        ]

        for i, (name, func) in enumerate(attitude_params):
            values = [func(state) for state in states]
            fig.add_trace(go.Scatter(x=times, y=values, name=name), row=i+1, col=2)
            fig.update_yaxes(title_text=name, row=i+1, col=2)

        # Update layout
        fig.update_layout(
            height=1000,
            width=1500,
            title_text="Integrated Orbit and Attitude Simulation",
            showlegend=False,
            title_x=0.5,
            title_y=0.99,
            title_xanchor='center',
            title_yanchor='top'
        )

        # Add column titles
        fig.add_annotation(
            x=0.25, y=1.05, xref="paper", yref="paper",
            text="Orbit Parameters", showarrow=False, font=dict(size=16)
        )
        fig.add_annotation(
            x=0.75, y=1.05, xref="paper", yref="paper",
            text="Attitude Parameters", showarrow=False, font=dict(size=16)
        )

        # Update x-axes to show time
        for i in range(7):
            fig.update_xaxes(title_text="Time (s)", row=i+1, col=1, showticklabels=True)
            fig.update_xaxes(title_text="Time (s)", row=i+1, col=2, showticklabels=True)

        # Adjust subplot titles
        for i, ann in enumerate(fig['layout']['annotations']):
            ann['font'] = dict(size=12)
            if i >= 14:  # Adjusting the position of column titles
                ann['y'] = 1.06

        fig.show()
    
def run():
    sim = IntegratedSimulation()
    states = sim.run_simulation()
    sim.plot_results(states)

if __name__ == "__main__":
    run()