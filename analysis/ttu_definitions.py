import os

SOURCE_DIRECTORY = 'C:/Users/22wal/Documents/GLWind/data/200-m MetTower Data/200m-2018-12/tower/procd/2018/Dec'
OUTPUT_DIRECTORY = 'C:/Users/22wal/Documents/GLWind/ttu_results/01dec18-15dec18/december2018_data'

# Local gravity at TTU (latitude ~ 33.59 degrees, elevation ~ 1014 m), in m/s^2
LOCAL_GRAVITY = 9.793

# Latitude and longitude of TTU 200-meter tower, each in degrees
LATITUDE = 33.59
LONGITUDE = -102.03
ELEVATION_METERS = 1014

SOURCE_TIMEZONE = 'UTC'
LOCAL_TIMEZONE = 'US/Central'

NPROC = os.cpu_count()

SOURCE_UNITS = {
    'p' : 'inHg',
    't' : 'F',
    'ts' : 'F',
    'rh' : '%',
    'u' : 'mph',
    'v' : 'mph',
    'w' : 'mph',
    'ws' : 'mph',
    'wd' : ('degrees', 'N', 'CW')
}

FIGVARS = {
    'p' : 'Barometric pressure',
    't' : 'Temperature',
    'ts' : 'Sonic temperature',
    'vpt' : 'Virtual potential temperature',
    'ws' : 'Wind speed',
    'u' : 'U wind component',
    'v' : 'V wind component',
    'w' : 'W wind component',
    'rh' : 'Relative humidity',
    "w'u'" : 'Vertical flux of u-momentum',
    "w'v'" : 'Vertical flux of v-momentum',
    "w'vpt'" : 'Vertical heat flux'
}

FIGUNITS = {
    'p' : 'kPa',
    't' : 'K',
    'ts' : 'K',
    'vpt' : 'K',
    'ws' : 'm/s',
    'u' : 'm/s',
    'v' : 'm/s',
    'w' : 'm/s',
    'rh' : 'decimal fraction',
    "w'u'" : 'm^2/s^2',
    "w'v'" : 'm^2/s^2',
    "w'vpt'" : 'K*m/s'
}

SOURCE_HEADERS = ['SU_1', 'SV_1', 'SW_1', 'TST_1', 'TT_1', 'TRH_1', 'TBP_1', 'SU_2', 'SV_2', 'SW_2', 'TST_2', 'TT_2', 'TRH_2', 'TBP_2', 'SU_3', 'SV_3', 'SW_3', 'TST_3', 'TT_3', 'TRH_3', 'TBP_3', 'SU_4', 'SV_4', 'SW_4', 'TST_4', 'TT_4', 'TRH_4', 'TBP_4', 'SU_5', 'SV_5', 'SW_5', 'TST_5', 'TT_5', 'TRH_5', 'TBP_5', 'SU_6', 'SV_6', 'SW_6', 'TST_6', 'TT_6', 'TRH_6', 'TBP_6', 'SU_7', 'SV_7', 'SW_7', 'TST_7', 'TT_7', 'TRH_7', 'TBP_7', 'SU_8', 'SV_8', 'SW_8', 'TST_8', 'TT_8', 'TRH_8', 'TBP_8', 'SU_9', 'SV_9', 'SW_9', 'TST_9', 'TT_9', 'TRH_9', 'TBP_9', 'SU_10', 'SV_10', 'SW_10', 'TST_10', 'TT_10', 'TRH_10', 'TBP_10', 'TSN-TRANS_1', 'TSNW-TRANS_1', 'TSV-TRANS_1', 'TS-WS_1', 'TS-WD_1', 'TSN-TRANS_2', 'TSNW-TRANS_2', 'TSV-TRANS_2', 'TS-WS_2', 'TS-WD_2', 'TSN-TRANS_3', 'TSNW-TRANS_3', 'TSV-TRANS_3', 'TS-WS_3', 'TS-WD_3', 'TSN-TRANS_4', 'TSNW-TRANS_4', 'TSV-TRANS_4', 'TS-WS_4', 'TS-WD_4', 'TSN-TRANS_5', 'TSNW-TRANS_5', 'TSV-TRANS_5', 'TS-WS_5', 'TS-WD_5', 'TSN-TRANS_6', 'TSNW-TRANS_6', 'TSV-TRANS_6', 'TS-WS_6', 'TS-WD_6', 'TSN-TRANS_7', 'TSNW-TRANS_7', 'TSV-TRANS_7', 'TS-WS_7', 'TS-WD_7', 'TSN-TRANS_8', 'TSNW-TRANS_8', 'TSV-TRANS_8', 'TS-WS_8', 'TS-WD_8', 'TSN-TRANS_9', 'TSNW-TRANS_9', 'TSV-TRANS_9', 'TS-WS_9', 'TS-WD_9', 'TSN-TRANS_10', 'TSNW-TRANS_10', 'TSV-TRANS_10', 'TS-WS_10', 'TS-WD_10']

HEADER_MAP = {
    'SU' : None,
    'SV' : None,
    'SW' : None,
    'TST' : 'ts',
    'TT' : 't',
    'TRH' : 'rh',
    'TBP' : 'p',
    'TSN-TRANS' : 'u',
    'TSNW-TRANS' : 'v',
    'TSV-TRANS' : 'w',
    'TS-WS' : 'ws',
    'TS-WD' : 'wd'
}

HEADER_MAP_INV = {v : k for k, v in HEADER_MAP.items()}

BOOMS_LIST = list(range(1, 11))
HEIGHTS_LIST = [0.9, 2.4, 4.0, 10.1, 16.8, 47.3, 74.7, 116.5, 118.2, 200.0]
HEIGHTS = {b : h for b, h in zip(BOOMS_LIST, HEIGHTS_LIST)}
