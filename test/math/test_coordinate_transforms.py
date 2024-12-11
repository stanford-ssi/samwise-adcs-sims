import numpy as np

from simwise.math.coordinate_transforms import mee_to_coe, coe_to_mee, ECEF_to_topocentric, topocentric_to_ECEF

def test_mee_to_coe_and_reverse():
    orbit_kep = np.array([7000e3, 0.1, 0.1, 0.1, 0.1, 0.1])
    orbit_mee = coe_to_mee(orbit_kep)
    orbit_kep_new = mee_to_coe(orbit_mee)
    assert np.allclose(orbit_kep, orbit_kep_new)

def ECEF_to_topocentric_and_reverse():
    r = np.array([1e7, 1e7, 1e7])
    r_topocentric = ECEF_to_topocentric(r)
    r_new = topocentric_to_ECEF(*r_topocentric)
    assert np.allclose(r, r_new)