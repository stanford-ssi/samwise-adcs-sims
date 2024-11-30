import asyncio
import aiohttp
import numpy as np
import json
import pickle
import os
import re

from simwise.utils.plots import plot3D, plotJD, plot_position_over_jd, plot_single
from simwise.math.frames import ecliptic_to_equatorial
from simwise.utils.time import julian_date_from_timestamp
from simwise.attitude_determination.sun import approx_sun_position
from simwise.math.axis_angle import get_axis_angle

CACHE_FILE = "earth_position_cache.pkl"

async def get_position_from_horizons(params):
    url = "https://ssd.jpl.nasa.gov/api/horizons.api"

    # Normalize the cache key by serializing params consistently
    cache_key = json.dumps(params, sort_keys=True)

    # Load cache if it exists
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as cache:
            cached_data = pickle.load(cache)
            # Check if the key exists in the cache
            if cache_key in cached_data:
                print("Cache hit!")
                return cached_data[cache_key]
    else:
        # Initialize an empty cache file
        with open(CACHE_FILE, "wb") as cache:
            pickle.dump({}, cache)

    print("Cache miss! Fetching data from the API...")

    # Call the API if no cache is found
    async def call_api(url, params):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.text()
                    result = parse_horizons_vector(data)

                    # Cache the result
                    with open(CACHE_FILE, "rb") as cache:
                        cached_data = pickle.load(cache)
                    cached_data[cache_key] = result
                    with open(CACHE_FILE, "wb") as cache:
                        pickle.dump(cached_data, cache)

                    print("Data cached successfully.")
                    return result
                else:
                    raise Exception(f"Failed to get data: {response.status}")

    return await call_api(url, params)

def parse_horizons_vector(response_text):
    # Split the response into lines
    lines = response_text.splitlines()
    vector_start = False
    positions = []
    times = []

    for line in lines:
        if line.strip() == "$$SOE":  # Start of vector data
            vector_start = True
            continue
        elif line.strip() == "$$EOE":  # End of vector data
            break
        elif vector_start:
            if "=" in line and "TDB" in line:
                # Parse the timestamp
                timestamp = line.split("=")[-1].strip()
                jd = julian_date_from_timestamp(timestamp)
                times.append(jd)  # Extract the time portion
            elif "X =" in line and "Y =" in line and "Z =" in line:
                # Parse the position vector
                match = re.search(r"X =\s*([-\d.E+]+)\s*Y =\s*([-\d.E+]+)\s*Z =\s*([-\d.E+]+)", line)
                if match:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    z = float(match.group(3))
                    position_vector = np.array([x, y, z])
                    positions.append(position_vector)
    
    return np.array(positions), np.array(times)


async def main():
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
    sun_positions, times = await get_position_from_horizons(params)
    sun_positions_normalized = sun_positions / np.linalg.norm(sun_positions, axis=1)[:, np.newaxis]

    # plot3D(sun_vector_eci)
    # plotJD(times, "jd.png")
    # plot_position_over_jd(sun_vector_eci, times)
    plot_position_over_jd(sun_positions_normalized, times)

    with open(CACHE_FILE, "rb") as cache:
        cached_data = pickle.load(cache)
        if cache_key in cached_data:
            positions, times = cached_data[cache_key]
            r_sun_approx = np.array([approx_sun_position(time) for time in times])
            
    # plot error
    errors = []
    for i in range(times.shape[0]):
        axis, angle = get_axis_angle(sun_positions_normalized[i], r_sun_approx[i])
        
        errors.append(np.rad2deg(angle))

    errors = np.array(errors)
    print(errors)
    plot_single(errors, times)
    

# Run the async main function
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    task = loop.create_task(main())
    loop.run_until_complete(task)