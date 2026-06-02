# test the range

import numpy as np
from simwise.utils.transforms import ecef2enu, blh2ecef, enu2azel, eci2ecef

blh_durand = np.array([np.radians(37.42687087), np.radians(-122.17333307), 48.0])
blh_mission_peak = np.array([np.radians(37.520414), np.radians(-121.875448), 618.0])

r_durand = blh2ecef(blh_durand)
r_mission_peak = blh2ecef(blh_mission_peak)
dr = r_mission_peak - r_durand

dr_enu = ecef2enu(r_durand, blh_mission_peak)
azimuth, elevation = enu2azel(dr_enu)

# to degrees
azimuth = np.degrees(azimuth)
elevation = np.degrees(elevation)

print(f"Distance: {np.linalg.norm(dr)} km")
print(f"Azimuth: {azimuth}, Elevation: {elevation}")