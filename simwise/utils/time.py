import re
import datetime
import numpy as np
import math

def jd_to_dt_utc(JD):
    Q = JD+0.5
    Z = np.floor(Q)
    W = np.floor((Z - 1867216.25)/36524.25)
    X = np.floor(W/4)
    A = Z+1+W-X
    B = A+1524
    C = np.floor((B-122.1)/365.25)
    D = np.floor(365.25*C)
    E = np.floor((B-D)/30.6001)
    F = np.floor(30.6001*E)
    day = np.floor(B-D-F+(Q-Z))
    if 1 <= (E-1) <= 12:
        month = E-1
    else:
        month = E-13
    
    if month <= 2:
        year = C-4715
    else:
        year = C-4716
    
    dt = datetime.datetime(int(year), int(month), int(day))
    fractional_days = days=JD - np.floor(JD)

    #XXX This is a hack to correct the time
    dt += datetime.timedelta(float(fractional_days) - 0.5)
    
    return dt


def dt_utc_to_jd(utc_time):
    """Convert UTC time to Julian date.

    This calculation is only valid for days after March 1900.

    Args:
        utc_time (datetime): The observation time as a datetime object

    Returns:
        julian_date (float): The observation time as a julian date.
    """
    year, month, day = utc_time.year, utc_time.month, utc_time.day
    julian_date = (
        367 * year - 7 * (year + (month + 9) // 12) // 4 + 275 * month // 9 + day + 1721013.5
    )

    # update with the frational day
    julian_date +=  (
        utc_time.hour + utc_time.minute / 60 
        + (utc_time.second + 1e-6 * utc_time.microsecond) / 3600 
    ) / 24

    return julian_date