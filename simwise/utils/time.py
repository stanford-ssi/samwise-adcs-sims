"""
Time utilities

@ Author: Lundeen Cahilly
@ Date: 2026-02-21
"""

import numpy as np
from simwise.satellite import SatelliteState
from simwise.constants import DEG2RAD

def mjd(state: SatelliteState):
    return state.mjd_epoch + state.t / 86400.0 # [days]

def gmst(state: SatelliteState):
    return mjd2gmst(mjd(state)) # [rad]

def mjd2gmst(mjd):
    d = mjd - 51544.5 # [days]
    gmst = 280.46061837 + 360.98564736629 * d # [deg]
    return gmst * DEG2RAD # [rad]

def date2mjd(Y, M, D):
    # source: satellite orbits by montebruck section a.1.1
    
    # leap year handling in MJD lets year run from march 1 to end of february (eqn a.4)
    if M <= 2:
        y = Y - 1
        m = M + 12
    else:
        y = Y
        m = M

    # compute auxilirary quantity B (eqn a.5)
    if Y < 1582 or (Y == 1582 and M < 10) or (Y == 1582 and M == 10 and D <= 4):
        B = -2 + np.floor((y + 4716) / 4) - 1179
    else:
        B = np.floor(y / 400) - np.floor(y / 100) + np.floor(y / 4)

    # compute modified Julian Date (MJD) (eqn a.6)
    mjd = 365*y - 679004 + np.floor(B) + np.floor(30.6001 * (m+1)) + D
    return mjd

def jd2mjd(jd):
    return jd - 2400000.5 # [days]

def mjd2jd(mjd):
    return mjd + 2400000.5 # [days]

def jd2gmst(jd):
    return mjd2gmst(mjd2jd(jd))