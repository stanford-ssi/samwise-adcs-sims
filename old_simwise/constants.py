# File to define all constants used in sims
import numpy as np

# Orbital Properties:
EARTH_RADIUS_M = 6378e3     # m
EARTH_J2 = 0.00108262672    # JGM2
MU_EARTH = 3.986e14         # m^3 s^-2

SATELLITE_ALTITUDE = 450e3  # m TODO - move to orbit

# Satellite Geometric Properties:
CUBESAT_HEIGHT = 0.20                   # m
CUBESAT_WIDTH = 0.10                    # m
SOLARPANEL_HEIGHT = CUBESAT_HEIGHT      # m
SOLARPANEL_WIDTH = CUBESAT_WIDTH * 2    # m

# Satellite Aerodyamic Properties:
# Defined looking at the satellite with the solar-panel deployed in max area config
MAX_WETTED_AREA = (2 * SOLARPANEL_WIDTH + CUBESAT_WIDTH * np.sqrt(2)) * SOLARPANEL_HEIGHT

# Time
TDB_TO_UTC_OFFSET = -69.184             # s
SECONDS_PER_DAY = 86400                 # s