"""
Constants for the world

@ Author: Lundeen Cahilly
@ Date: 2026-02-20
"""
import numpy as np

# Earth constants
MU_EARTH = 3.986004418e14 # [m^3/s^2]
R_EARTH = 6378137.0 # [m] WGS-84 equatorial radius
e_wgs84 = 0.0818191908426 # eccentricity

# Math constants
rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0