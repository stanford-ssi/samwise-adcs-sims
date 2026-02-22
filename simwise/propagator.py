from tqdm import tqdm
from simwise.dynamics.eom import orbit_dot, attitude_dot
from simwise.math.integrators import rk4

def propagate(state, params, torques: list[callable], perturbations: list[callable], dt: float = 0.1, orbit_every: int = 100, tf: float = 100.0):
    trajectory = []
    f_orbit = lambda s, t: orbit_dot(s, params, perturbations=perturbations)
    f_attitude = lambda s, t: attitude_dot(s, params, torques=torques)

    mjd_epoch = state.mjd_epoch
    n_steps = int(tf / dt)
    step = 0
    with tqdm(total=n_steps, desc="Propagating", unit="step") as pbar:
        while state.t < tf:
            trajectory.append(state)
            if step % orbit_every == 0:  # propagate orbit a lot less often than attitude
                state_orbit = rk4(state, orbit_every * dt, f_orbit)
            state = rk4(state, dt, f_attitude)
            state.r = state_orbit.r
            state.v = state_orbit.v
            state.mjd_epoch = mjd_epoch
            step += 1
            pbar.update(1)

    return trajectory
