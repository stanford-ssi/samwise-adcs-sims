import re
import datetime

def julian_date_from_timestamp(timestamp):
    """
    Convert a timestamp like 'A.D. 2024-Nov-29 00:00:00.0000 TDB' to Julian Date.
    """
    # Extract the date and time components
    match = re.search(r"(\d{4})-(\w{3})-(\d{2}) (\d{2}:\d{2}:\d{2})", timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    year, month_str, day, time_str = match.groups()

    # Convert month abbreviation to number
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    month = month_map[month_str]

    # Parse the time
    hour, minute, second = map(int, time_str.split(":"))

    # Convert to datetime
    dt = datetime.datetime(int(year), month, int(day), hour, minute, second)

    # Convert to Julian Date
    JD = dt.toordinal() + 1721424.5 + (hour + minute / 60 + second / 3600) / 24
    return JD