import json
import pickle
import asyncio
import numpy as np
import os
from datetime import datetime, timedelta
from simwise.world_model.sun import approx_sun_position
from simwise.math.axis_angle import vectors_to_axis_angle
from simwise.utils.horizons import get_position_from_horizons, CACHE_FILE
from simwise.utils.plots import plot_subplots

def test_approx_sun_position():
    params = {
        "format": "text",
        "COMMAND": "10",  # Earth ID
        "CENTER": "500@399",  # Sun-centered frame
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "START_TIME": "2024-1-1",  # Current date
        "STOP_TIME": "2025-1-1",
        "STEP_SIZE": "1d",  # Corrected step size
        "OUT_UNITS": "KM-S",
        "REF_PLANE": "FRAME",
        "REF_SYSTEM": "J2000"
    }
    cache_key = json.dumps(params, sort_keys=True)

    # Get Earth's position in J2000 frame relative to the Sun
    sun_positions, times = get_position_from_horizons(params)
    sun_positions_normalized = sun_positions / np.linalg.norm(sun_positions, axis=1)[:, np.newaxis]

    plot_subplots(times, sun_positions_normalized, ["X", "Y", "Z"], "Time [s]", "Sun Position from Horizons", save=False)

    with open(CACHE_FILE, "rb") as cache:
        cached_data = pickle.load(cache)
        if cache_key in cached_data:
            positions, times = cached_data[cache_key]
            r_sun_approx = np.array([approx_sun_position(time) for time in times])
            
    # plot error
    errors = []
    for i in range(times.shape[0]):
        axis, angle = vectors_to_axis_angle(sun_positions_normalized[i], r_sun_approx[i])
        
        errors.append(np.rad2deg(angle))

    errors = np.array(errors)
    print(times.shape, errors.shape)
    plot_subplots(times, errors, ["Error [°]"], "Time [s]", "Approx Sun Position Error", save=False)
    

# # Run the async main function
# if __name__ == "__main__":
