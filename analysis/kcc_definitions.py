import pandas as pd

# Timezones (source: data, local: local timezone of collection location)
SOURCE_TIMEZONE = 'UTC'
LOCAL_TIMEZONE = 'US/Central'

# Start and end times of data (for storm and precipitation matching)
START_TIME = pd.to_datetime('2017-09-21 19:00:00-05:00')
END_TIME = pd.to_datetime('2018-08-29 00:30:00-05:00')

# Local gravity at Cedar Rapids (latitude ~ 42 degrees, elevation ~ 247 m), in m/s^2
LOCAL_GRAVITY = 9.802

# Latitude and longitude of KCC met tower, each in degrees
LATITUDE = 41.91
LONGITUDE = -91.65
ELEVATION_METERS = 247

SOURCE_UNITS = {
        'p' : 'mmHg',
        't' : 'C',
        'rh' : '%',
        'ws' : 'm/s',
        'wd' : ['degrees', 'W', 'CW'],
}

# All heights (in m) that data exists at
# Data columns will (and must) follow '{type}_{boom}' format
HEIGHT_LIST = [6., 10., 20., 32., 80., 106.]
BOOM_LIST = [1, 2, 3, 4, 5, 6]
HEIGHTS = {b : h for b, h in zip(BOOM_LIST, HEIGHT_LIST)}

STORM_FILES = [
    "C:/Users/22wal/Documents/GLWind/data/StormEvents/StormEvents_details-ftp_v1.0_d2017_c20250122.csv",
    "C:/Users/22wal/Documents/GLWind/data/StormEvents/StormEvents_details-ftp_v1.0_d2018_c20240716.csv"
]

CID_DATA_PATH = "C:/Users/22wal/Documents/GLWind/data/CID/CID_Sep012017_Aug312018.csv"

CID_UNITS = {
        'p' : f'mBar_{ELEVATION_METERS}asl',
        't' : 'C',
        'rh' : '%',
        'ws' : 'mph',
        'wd' : ['degrees', 'N', 'CW'],
}

CID_TRACE = 0.0001
