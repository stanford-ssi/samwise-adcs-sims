import erfa
import numpy as np

def ecef_to_eci_IAU(eci_vector, dt_utc):
    # Compute UTC Julian Date
    jd_utc = erfa.dtf2d("UTC", dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second)
    
    # Convert UTC to TAI
    tai1, tai2 = erfa.utctai(jd_utc[0], jd_utc[1])
    
    # Convert TAI to TT
    tt1, tt2 = erfa.taitt(tai1, tai2)
    
    # For accurate results, replace dut1 with the actual UT1-UTC value in seconds
    dut1 = 0.0  # UT1 - UTC in seconds
    
    # Convert UTC to UT1
    ut11, ut12 = erfa.utcut1(jd_utc[0], jd_utc[1], dut1)
    
    # Polar motion coordinates in radians (replace with actual values for higher accuracy)
    xp = 0.0
    yp = 0.0
    
    # Compute celestial-to-terrestrial transformation matrix
    rc2t = erfa.c2t00b(tt1, tt2, ut11, ut12, xp, yp)
    
    # Transform ECI vector to ECEF
    return np.dot(rc2t, eci_vector)