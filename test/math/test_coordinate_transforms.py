import numpy as np

from simwise.math.coordinate_transforms import mee_to_coe, coe_to_mee

def test_mee_to_coe_and_reverse():
    orbit_kep = np.array([7000e3, 0.1, 0.1, 0.1, 0.1, 0.1])
    orbit_mee = coe_to_mee(orbit_kep)
    orbit_kep_new = mee_to_coe(orbit_mee)
    assert np.allclose(orbit_kep, orbit_kep_new)