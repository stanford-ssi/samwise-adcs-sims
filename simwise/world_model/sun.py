import numpy as np
from simwise import constants

def approx_sun_position(JD):
    # convert to days after J2000
    D = JD - 2451545.0
    g = np.deg2rad((357.529 + (0.98560028 * D)) % 360) # radians
    # r_sun = (1.00014 - (0.01671*np.cos(g)) - (0.00014*np.cos(2 * g))) * 149597870.7 * 1000
    q = 280.459 + (0.98564736*D)
    λ_sun = q + (1.915*np.sin(g)) + (0.020*np.sin(2. * g))
    λ_sun_rad = np.deg2rad(λ_sun)
    ε = 23.439 - (0.00000036*D)
    ε_rad = np.deg2rad(ε)

    # rotate to ECI by rotation of ecliptic to equatorial plane
    unit_vector = [np.cos(λ_sun_rad), np.sin(λ_sun_rad)*np.cos(ε_rad), np.sin(λ_sun_rad)*np.sin(ε_rad)]
    return(unit_vector)

def eclipse_model(r_sun, r_sat):
    # calculate the angle between the sun and the satellite
    sat_unit_vector = r_sat / np.linalg.norm(r_sat)
    Ψ = np.arccos(np.dot(r_sun, sat_unit_vector))
    # calculate distance into the cone of the shadow
    a = np.linalg.norm(r_sat) * np.sin(Ψ)

    return Ψ > np.pi/2 and a < constants.EARTH_RADIUS_M
    