import pandas as pd
import numpy as np
from windprofiles.classify import CoordinateRegion

ALL_COLUMNS = ['BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME', 'END_YEARMONTH',
       'END_DAY', 'END_TIME', 'EPISODE_ID', 'EVENT_ID', 'STATE', 'STATE_FIPS',
       'YEAR', 'MONTH_NAME', 'EVENT_TYPE', 'CZ_TYPE', 'CZ_FIPS', 'CZ_NAME',
       'WFO', 'BEGIN_DATE_TIME', 'CZ_TIMEZONE', 'END_DATE_TIME',
       'INJURIES_DIRECT', 'INJURIES_INDIRECT', 'DEATHS_DIRECT',
       'DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'SOURCE',
       'MAGNITUDE', 'MAGNITUDE_TYPE', 'FLOOD_CAUSE', 'CATEGORY', 'TOR_F_SCALE',
       'TOR_LENGTH', 'TOR_WIDTH', 'TOR_OTHER_WFO', 'TOR_OTHER_CZ_STATE',
       'TOR_OTHER_CZ_FIPS', 'TOR_OTHER_CZ_NAME', 'BEGIN_RANGE',
       'BEGIN_AZIMUTH', 'BEGIN_LOCATION', 'END_RANGE', 'END_AZIMUTH',
       'END_LOCATION', 'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON',
       'EPISODE_NARRATIVE', 'EVENT_NARRATIVE', 'DATA_SOURCE']
COLUMNS_OF_INTEREST = ['STATE', 'EVENT_TYPE',
       'BEGIN_DATE_TIME', 'END_DATE_TIME', 'BEGIN_LOCATION', 'END_LOCATION',
       'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON',
       'EPISODE_NARRATIVE', 'EVENT_NARRATIVE',
       'CZ_TIMEZONE',]

def get_storms(filepath: str, region: CoordinateRegion):
    df = pd.read_csv(filepath)
    columns_to_drop = [column for column in ALL_COLUMNS if column not in COLUMNS_OF_INTEREST]
    df.drop(columns = columns_to_drop, inplace = True)
    classifier = lambda row : region.classify(row['BEGIN_LAT'], row['BEGIN_LON']) or region.classify(row['END_LAT'], row['END_LON'])
    df = df[df.apply(classifier, axis = 1)]
    return df
