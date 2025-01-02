import pandas as pd
import numpy as np
import preprocess
import compute
from classify import TerrainClassifier, SingleClassifier

# Where to pull data from
data_directory = '../../data/KCC_SlowData'

# Time periods during which data is untrustworthy and should be removed
REMOVAL_PERIODS = {
    ('2018-03-05 13:20:00','2018-03-10 00:00:00') : 'ALL', # large maintenance gap
    ('2018-04-18 17:40:00','2018-04-19 14:20:00') : [106], # small maintenance-shaped gap
    ('2018-09-10 12:00:00','2018-09-20 12:00:00') : 'ALL' # blip at end
}

# Define the terrain classification scheme
TERRAIN_CLASSIFIER = TerrainClassifier(complexCenter = 315,
                                          openCenter = 135,
                                          radius = 15,
                                          inclusive = True,
                                          height = 10)

# Define 3-class bulk Richardson number stability classification scheme
STABILITY_CLASSIFIER = SingleClassifier(parameter = 'Ri_bulk')
STABILITY_CLASSIFIER.add_class('unstable', '(-inf,-0.1)')
STABILITY_CLASSIFIER.add_class('neutral', '[-0.1,0.1]')
STABILITY_CLASSIFIER.add_class('stable', '(0.1,inf)')

INSTRUCTIONS = {
    'terrain_classifier' :  TERRAIN_CLASSIFIER,
    #'stability_classifier' : STABILITY_CLASSIFIER,
}

# All heights (in m) that data exists at
# Data columns will (and must) follow '{type}_{height}m' format
HEIGHTS = [6, 10, 20, 32, 80, 106]

# Read in the data from the booms and set column names to common format
boom1 = pd.read_csv(f'{data_directory}/Boom1OneMin.csv').rename(columns={'TimeStamp' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_6m',
                                                       'MeanDirection' : 'wd_6m',
                                                       'MeanTemperature (C )' : 't_6m',
                                                       'MeanPressure (mmHg)' : 'p_6m'})

boom2 = pd.read_csv(f'{data_directory}/Boom2OneMin.csv').rename(columns={'TIMESTAMP' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_10m',
                                                       'MeanDirection' : 'wd_10m',
                                                       'MeanTemperature (C )' : 't_10m',
                                                       'MeanRH (%)' : 'rh_10m'})

boom3 = pd.read_csv(f'{data_directory}/Boom3OneMin.csv').rename(columns={'TIMESTAMP' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_20m',
                                                       'MeanDirection' : 'wd_20m'})

boom4 = pd.read_csv(f'{data_directory}/Boom4OneMin.csv').rename(columns={'TimeStamp' : 'time',
                                                       'MeanVelocity' : 'ws_32m',
                                                       'MeanDirection' : 'wd_32m',
                                                       'MeanTemperature' : 't_32m',
                                                       'MeanRH' : 'rh_32m'})

boom5 = pd.read_csv(f'{data_directory}/Boom5OneMin.csv').rename(columns={'TimeStamp' : 'time',
                                                       'MeanVelocity' : 'ws_80m',
                                                       'MeanDirection' : 'wd_80m',
                                                       'MeanTemperature' : 't_80m',
                                                       'MeanRH' : 'rh_80m'})

boom6 = pd.read_csv(f'{data_directory}/Boom6OneMin.csv').rename(columns={'TIMESTAMP' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_106m1',
                                                       'Mean Direction' : 'wd_106m1',
                                                       'MeanTemperature (C )' : 't_106m',
                                                       'MeanRH (%)' : 'rh_106m'})

boom7 = pd.read_csv(f'{data_directory}/Boom7OneMin.csv').rename(columns={'TimeStamp' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_106m2',
                                                       'MeanDirection' : 'wd_106m2',
                                                       'MeanPressure (mmHg)' : 'p_106m'})

# Merge the data together into one pd.DataFrame
boom_list = [boom1,boom2,boom3,boom4,boom5,boom6,boom7]
for boom in boom_list:
    boom['time'] = pd.to_datetime(boom['time'])
    boom.set_index('time', inplace=True)
df = (boom1.merge(boom2, on='time', how='inner')
        .merge(boom6, on='time', how='inner')
        .merge(boom7, on='time', how='inner')
        .merge(boom3, on='time',how='left')
        .merge(boom4, on='time', how='left')
        .merge(boom5, on='time', how='left')
        ) # maybe revisit and do outer only?

# Units that source data is in
original_units = {
    'p' : 'mmHg',
    't' : 'C',
    'rh' : '%',
    'ws' : 'm/s',
    'wd' : ['degrees', 'E', 'CW'],
}

# Automatically convert units of all columns to standards as outlined in convert.py
df = preprocess.convert_dataframe_units(df = df, from_units = original_units)

# Conditionally merge wind data from booms 6 and 7
# Boom 6 (106m1, west side) is shadowed near 90 degrees (wind from east)
# Boom 7 (106m2, east side) is shadowed near 270 degrees (wind from west)
# Important that we do this after the conversion step, to make sure wind angles are correct
df['ws_106m'], df['wd_106m'] = preprocess.shadowing_merge(
    df = df,
    speeds = ['ws_106m1', 'ws_106m2'],
    directions = ['wd_106m1', 'wd_106m2'],
    angles = [90, 270],
    width = 30, # Winds from within 30/2=15 degrees of tower are discarded
    drop_old = True # Discard the 106m1 and 106m2 columns afterwards
)

# Final common formatting changes:
# ws = 0 --> wd = pd.nan; types --> float32; sorting & duplication fixes
df = preprocess.clean_formatting(df = df, type = 'float32')

# Remove data according to REMOVAL_PERIODS
df = preprocess.remove_data(df = df, periods = REMOVAL_PERIODS)

# Rolling outlier removal
df = preprocess.rolling_outlier_removal(df = df,
                                        window_size_minutes = 30,
                                        sigma = 5,
                                        column_types = ['ws', 't', 'p', 'rh']) # maybe revisit and only remove wind speed outliers?

# Resampling into 10 minute intervals
df = preprocess.resample(df = df,
                         window_size_minutes = 10,
                         how = 'mean',
                         all_heights = HEIGHTS)

df = compute.compute_values(df = df, instructions = INSTRUCTIONS)

###
# TEMPORARY: for comparison to old results
dfOld = pd.read_csv('../../outputs/slow/combined.csv')
dfOld['time'] = pd.to_datetime(dfOld['time'])
dfOld.set_index('time', inplace=True)
dfOld = preprocess.clean_formatting(dfOld, silent = True) # for consistency in comparison
df.to_csv('../../results/combined.csv')
dfOld.to_csv('../../results/OLD.csv')
# from lib.other import test_frame_discrepancy_by_row
# test_frame_discrepancy_by_row(df, dfOld, progress = True) # Used this to find the ws_106m 2x error
###
