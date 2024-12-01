from datetime import timedelta, datetime

import erfa
import numpy as np

from simwise.math import frames
from simwise.utils.time import dt_utc_to_jd
from simwise.utils.plots import plot_subplots

"""
PyERFA is the Python wrapper for the ERFA library (Essential Routines for 
Fundamental Astronomy), a C library containing key algorithms for astronomy, 
which is based on the SOFA library published by the International Astronomical 
Union (IAU).
"""
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
    
    # Polar motion coordinates in radians (TODO replace with actual values for higher accuracy)
    xp = 0.0
    yp = 0.0
    
    # Compute celestial-to-terrestrial transformation matrix
    rc2t = erfa.c2t00b(tt1, tt2, ut11, ut12, xp, yp)
    
    # Transform ECI vector to ECEF
    return np.dot(rc2t, eci_vector)

def test_display_differences_in_conversion_methods():
    reference_vector = np.array([7000, 0, 0])
    diff_geneci = []
    diff_simwise = []
    diff_geneci_table = []
    diff_geneci_rotation_only = []
    year_pn_table = []

    for year in range(30):
        for day in range(365):
            dt = datetime(2000 + year, 1, 1) + timedelta(days=day)
            ecef_vector_IAU = ecef_to_eci_IAU(reference_vector, dt)
            jd_geneci = dt_utc_to_jd(dt)
            ecef_vector_geneci = frames.eci_to_ecef(reference_vector, dt)
            diff_geneci.append(ecef_vector_IAU - ecef_vector_geneci)
            rot_mat = frames.rotation_matrix(jd_geneci)
            if day == 0:
                pn_matrix = frames.precession_nutation_matrix(jd_geneci)
                # print(pn_matrix)
                year_pn_table.append(pn_matrix)
            diff_geneci_table.append(
                ecef_vector_IAU - ((pn_matrix @ rot_mat).T @ reference_vector)
            )
            diff_geneci_rotation_only.append(
                ecef_vector_IAU - (rot_mat.T @ reference_vector)
            )

    diff_geneci = np.array(diff_geneci)
    diff_simwise = np.array(diff_simwise)
    diff_geneci_table = np.array(diff_geneci_table)
    diff_geneci_rotation_only = np.array(diff_geneci_rotation_only)

    plot_subplots(np.arange(diff_geneci.shape[0]), diff_geneci, ["x", "y", "z"], "days since J2000", "Difference between ECEF to ECI conversion methods [geneci]")
    plot_subplots(np.arange(diff_geneci.shape[0]), diff_geneci_table, ["x", "y", "z"], "days since J2000", "Difference between ECEF to ECI conversion methods [rotation only with table for nutation/precession]")
    plot_subplots(np.arange(diff_geneci.shape[0]), diff_geneci_rotation_only, ["x", "y", "z"], "days since J2000", "Difference between ECEF to ECI conversion methods [rotation only]")

def test_ecef_pn_matrix():
    jd = dt_utc_to_jd(datetime(2021, 1, 1))
    t_end_seconds = 20 * 24 * 3600
    pn_table = frames.generate_ecef_pn_table(jd, t_end_seconds)
    assert len(pn_table) == 3
    