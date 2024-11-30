import asyncio
import aiohttp
import numpy as np
import json
import pickle
import os
import re
from datetime import datetime, timedelta

from simwise.math.frames import ecliptic_to_equatorial
from simwise.utils.time import dt_utc_to_jd
from simwise import constants

CACHE_FILE = "earth_position_cache.pkl"

def get_position_from_horizons(params):
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

    loop = asyncio.get_event_loop()
    task = loop.create_task(call_api(url, params))
    loop.run_until_complete(task)

    return task.result()

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
                time_string = line.split("=")[-1].strip()
                
                # Parse the time string to extract date and time
                parsed_time = datetime.strptime(time_string.split(' ')[1], '%Y-%b-%d %H:%M:%S.%f')

                # Step 2: Approximate TDB to UTC offset
                tdb_to_utc_offset = timedelta(seconds=constants.TDB_TO_UTC_OFFSET)

                # Step 3: Apply the offset
                dt_utc = parsed_time + tdb_to_utc_offset

                jd = dt_utc_to_jd(dt_utc)
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