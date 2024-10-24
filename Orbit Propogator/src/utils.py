import math
import numpy as np
from datetime import datetime

from .constants import earth_radius

# Modified from Matlab code in Canvas
def cal_2_gmst(Y1, M1, D1, D):
    # Compute modified month and year
    if M1 <= 2:
        Y2 = Y1 - 1
        M2 = M1 + 12
    else:
        Y2 = Y1
        M2 = M1

    B = Y1 / 400 - Y1 / 100 + Y1 / 4

    # Decimal days
    D2 = D1 + D

    # Modified Julian Date
    MJD = 365 * Y2 - 679004 + int(B) + int(30.6001 * (M2 + 1)) + D2
    d = MJD - 51544.5

    # GMST in degrees
    GMST = 280.4606 + 360.9856473 * d
    GMST = math.radians(GMST) # Convert to radians
    GMST = GMST % (2 * math.pi) # Ensure GMST is in the range [0, 2 * pi)

    return GMST

def get_gmst_from_epoch(t: int) -> float:
    # Get the date and time from the epoch
    epoch = datetime.fromtimestamp(t)

    # Get the year, month, day, hour, minute, and second from the epoch
    year = epoch.year
    month = epoch.month
    day = epoch.day
    hour = epoch.hour
    minute = epoch.minute
    second = epoch.second

    # Get the GMST from the epoch
    gmst = cal_2_gmst(year, month, day, (hour * 3600 + minute * 60 + second)/86400)
    return gmst

def in_eclipse(stateSatellite: np.ndarray, stateSun: np.ndarray) -> bool:
    stateEclipse = np.zeros(len(stateSatellite))
    sunDotSatArr = np.zeros(len(stateSatellite))
    perpNormArr = np.zeros(len(stateSatellite))

    for i in range(len(stateSatellite)):
        # Get the satellite and sun position vectors
        satelliteVector = stateSatellite[i,:3]
        sunVector = stateSun[i,:3]

        # Get the projection of the satellite position vector onto the sun position vector
        sunDotSat = np.dot(sunVector, satelliteVector)

        # Get perpendicular and parallel components of the satellite position vector relative to the sun position vector
        sunUnitVector = sunVector / np.linalg.norm(sunVector)
        satelliteParallelSun = sunUnitVector * np.dot(sunUnitVector, satelliteVector)
        satellitePerpendicularSun = satelliteVector - satelliteParallelSun

        inEclipse = (sunDotSat < 0) and (np.linalg.norm(satellitePerpendicularSun) < earth_radius)
        stateEclipse[i] = inEclipse
        sunDotSatArr[i] = sunDotSat
        perpNormArr[i] = np.linalg.norm(satellitePerpendicularSun)

    return stateEclipse, sunDotSatArr, perpNormArr