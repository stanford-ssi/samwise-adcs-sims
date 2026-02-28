from tqdm import tqdm
from simwise.dynamics.eom import orbit_dot, attitude_dot
from simwise.math.integrators import rk4


def propagate(satellite, torques: list[callable], perturbations: list[callable], dt: float = 0.1, orbit_every: int = 100, tf: float = 100.0):
    """Propagates a Satellite in place, storing history on satellite.history. Returns satellite."""
    state = satellite.state
    params = satellite.params

    f_orbit = lambda s, t: orbit_dot(s, params, perturbations=perturbations)
    f_attitude = lambda s, t: attitude_dot(s, params, torques=torques)

    mjd_epoch = state.mjd_epoch
    n_steps = int(tf / dt)
    step = 0
    satellite.history = [state]

    with tqdm(total=n_steps, desc="Propagating", unit="step") as pbar:
        while state.t < tf:
            if step % orbit_every == 0:
                state_orbit = rk4(state, orbit_every * dt, f_orbit)
            state = rk4(state, dt, f_attitude)
            state.r = state_orbit.r
            state.v = state_orbit.v
            state.mjd_epoch = mjd_epoch
            satellite.history.append(state)

            if satellite.estimator is not None:
                measurements = satellite.read_sensors()
                satellite.estimator.update(measurements, dt)
                satellite.est_history.append(satellite.estimator.state)

            step += 1
            pbar.update(1)

    satellite.state = state
    return satellite


def propagate_batch(satellites, torques: list[callable], perturbations: list[callable], dt: float = 0.1, orbit_every: int = 100, tf: float = 100.0):
    for satellite in satellites:
        propagate(satellite, torques, perturbations, dt, orbit_every, tf)
    return satellites
