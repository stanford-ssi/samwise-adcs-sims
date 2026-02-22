from simwise.world.sun import sun_vector_eci
from simwise.satellite.state import SatelliteState
from simwise.math.quaternion import Quaternion
from simwise.utils.time import mjd, date2mjd
import numpy as np

def test_sun_vector():
    state = SatelliteState(
        q_eci2body=Quaternion(0, 0, 0, 1),
        w_eci=np.array([0.0, 0.0, 0.0]),
        r_eci=np.array([0.0, 0.0, 0.0]),
        v_eci=np.array([0.0, 0.0, 0.0]),
        mjd_epoch=date2mjd(2026, 3, 20) + 14/24 + 45/3600 + 2/86400, # vernal equinox
    )
    sun_eci = sun_vector_eci(state)
    sun_eci = sun_eci / np.linalg.norm(sun_eci)
    assert np.allclose(sun_eci, np.array([1.0, 0.0, 0.0]), atol=1e-2)
    print(sun_eci)
