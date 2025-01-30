import pandas as pd
import numpy as np
import windprofiles.preprocess as preprocess
import windprofiles.compute as compute
import windprofiles.storms as storms
from windprofiles.classify import TerrainClassifier, StabilityClassifier, CoordinateRegion
from windprofiles.analyze import save
import sys
import os

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
        'wd' : ['degrees', 'E', 'CW'],
}

# All heights (in m) that data exists at
# Data columns will (and must) follow '{type}_{height}m' format
HEIGHTS = [6, 10, 20, 32, 80, 106]

STORM_FILES = [
    "C:/Users/22wal/OneDrive/GLWind/data/StormEvents/StormEvents_details-ftp_v1.0_d2017_c20250122.csv",
    "C:/Users/22wal/OneDrive/GLWind/data/StormEvents/StormEvents_details-ftp_v1.0_d2018_c20240716.csv"
]

CID_DATA_PATH = "C:/Users/22wal/OneDrive/GLWind/data/CID/CID_Sep012017_Aug312018.csv"

CID_UNITS = {
        'p' : f'mBar_{ELEVATION_METERS}asl',
        't' : 'C',
        'rh' : '%',
        'ws' : 'mph',
        'wd' : ['degrees', 'N', 'CW'],
}

CID_TRACE = 0.0001

def load_data(data_directory: str, outer_merges: bool = False):
    print('START DATA LOADING')
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
    df = (boom1.merge(boom2, on='time', how='outer' if outer_merges else 'inner')
            .merge(boom6, on='time', how='outer' if outer_merges else 'inner')
            .merge(boom7, on='time', how='outer' if outer_merges else 'inner')
            .merge(boom3, on='time',how='outer' if outer_merges else 'left')
            .merge(boom4, on='time', how='outer' if outer_merges else 'left')
            .merge(boom5, on='time', how='outer' if outer_merges else 'left')
            )

    print("END DATA LOADING")

    return df.reset_index()
    
def perform_preprocessing(df,
                          shadowing_width,
                          removal_periods, # These will be passed in UTC so we will do the removal before converting data time to local US/Central
                          outlier_window,
                          outlier_sigma,
                          resampling_window,
                          storm_events = None,
                          weather_data = None,
                          storm_removal = None):
    print("START DATA PREPROCESSING")
    
    doWeather = not(storm_events is None or weather_data is None or storm_removal is None)

    # Automatically convert units of all columns to standards as outlined in convert.py
    df = preprocess.convert_dataframe_units(df = df, from_units = SOURCE_UNITS, gravity = LOCAL_GRAVITY)

    # Conditionally merge wind data from booms 6 and 7
    # Boom 6 (106m1, west side) is shadowed near 90 degrees (wind from east)
    # Boom 7 (106m2, east side) is shadowed near 270 degrees (wind from west)
    # Important that we do this after the conversion step, to make sure wind angles are correct
    df['ws_106m'], df['wd_106m'] = preprocess.shadowing_merge(
        df = df,
        speeds = ['ws_106m1', 'ws_106m2'],
        directions = ['wd_106m1', 'wd_106m2'],
        angles = [90, 270],
        width = shadowing_width, # Winds from within 30/2=15 degrees of tower are discarded
        drop_old = True # Discard the 106m1 and 106m2 columns afterwards
    )

    # Final common formatting changes:
    # ws = 0 --> wd = pd.nan; types --> float32; sorting & duplication fixes
    df = preprocess.clean_formatting(df = df, type = 'float32')

    # Remove data according to REMOVAL_PERIODS
    if removal_periods is not None:
        df = preprocess.remove_data(df = df, periods = removal_periods)

    # Convert time index from UTC to local time
    df = preprocess.convert_timezone(df = df,
                                    source_timezone = SOURCE_TIMEZONE,
                                    target_timezone = LOCAL_TIMEZONE)

    # Rolling outlier removal
    df = preprocess.rolling_outlier_removal(df = df,
                                            window_size_minutes = outlier_window,
                                            sigma = outlier_sigma,
                                            column_types = ['ws', 't', 'p', 'rh'])

    # Resampling into 10 minute intervals
    df = preprocess.resample(df = df,
                            window_size_minutes = resampling_window,
                            how = 'mean',
                            all_heights = HEIGHTS,)
                            # deviations = [f'ws_{h}m' for h in HEIGHTS])

    if doWeather:
        df = preprocess.determine_weather(df = df,
                storm_events = storm_events,
                weather_data = cid_data,
                trace_float = CID_TRACE)

    # Remove data where it is too stormy to rely on data
    df = preprocess.flagged_removal(df = df,
                                    flags = storm_removal)

    # Remove rows where there isn't enough data (not enough columns, or missing either 10m or 106m data)
    df = preprocess.strip_missing_data(df = df,
                                    necessary = [10, 106],
                                    minimum = 3)

    print("END DATA PREPROCESSING")

    return df

def compute_values(df,
                   terrain_classifier,
                   stability_classifier):

    print("BEGIN COMPUTATIONS")

    df = compute.virtual_potential_temperatures(df = df,
                                                heights = [10, 106],
                                                substitutions = {'p_10m' : 'p_6m'})
        
    df = compute.environmental_lapse_rate(df = df,
                                          variable = 'vpt',
                                          heights = [10, 106])
    
    df = compute.bulk_richardson_number(df = df,
                                        heights = [10, 106],
                                        gravity = LOCAL_GRAVITY)
        
    df = compute.classifications(df = df,
                                 terrain_classifier = terrain_classifier,
                                 stability_classifier = stability_classifier)
    
    df = compute.power_law_fits(df = df,
                                heights = HEIGHTS,
                                columns = [None,'alpha'])
    
    df = compute.power_law_fits(df = df,
                                heights = [6, 10, 20, 32, 80],
                                columns = [None,'alpha_no106'])
    
    df = compute.power_law_fits(df = df,
                                heights = [6, 10, 20, 80, 106],
                                columns = [None,'alpha_no32'])
    
    df = compute.strip_failures(df = df,
                                subset = ['Ri_bulk','alpha'])
        
    print("END COMPUTATIONS")

    return df

def temp_plots(df):
    import finalplots
    finalplots.generate_plots(df)
    #import ipplot as plot
    #plot.hist_alpha_by_stability(df, separate = True, compute = True, overlay = True)
    #plot.alpha_tod_violins(df, fit = False)

    # tod_dir = 'C:/Users/22wal/OneDrive/Pictures/temp/tods_wider'
    # plot.alpha_tod_violins(df, fit = True, saveto = f'{tod_dir}/year.png')
    # plot.alpha_tod_violins_by_terrain(df, fit = True, saveto = f'{tod_dir}/yearT.png')
    # for season in ['Fall', 'Winter', 'Spring', 'Summer']:
    #     plot.alpha_tod_violins(df, season = season, fit = True, saveto = f'{tod_dir}/{season.lower()}.png')
    #     plot.alpha_tod_violins_by_terrain(df, season = season, fit = True, saveto = f'{tod_dir}/{season.lower()}T.png') 

    #plot.ri_tod_violins(df, fit = False, cut = 25, printcutfrac = True, bounds = (-5,3))

    #plot.boom_data_available(df, freq = '10min', heights = HEIGHTS)

    #plot.alpha_over_time(df)
    #plot.comparison(df, which = ['alpha', 'alpha_no106'], xlims=(-0.5,1), ylims = (-0.5,1))
    #plot.comparison(df, which = ['alpha', 'alpha_no32'], xlims=(-0.5,1), ylims = (-0.5,1))



    return

def get_storm_events(start_time, end_time, radius: int|float = 25., unit: str = 'km'):
    region = CoordinateRegion(latitude = LATITUDE, longitude = LONGITUDE, radius = radius, unit = unit)
    results = []
    for filepath in STORM_FILES:
        results.append(storms.get_storms(filepath, region))
    sDf = pd.concat(results)
    # Checking sDf['CZ_TIMEZONE'] shows that all are localized to CST-6 regardless of DST
    #   To account for this in our localization we specify GMT-6
    format_string = "%d-%b-%y %H:%M:%S" # Format can't be inferred so specifying. See strftime documentation.
    sDf['BEGIN_DATE_TIME'] = pd.to_datetime(sDf['BEGIN_DATE_TIME'], format = format_string).dt.tz_localize('etc/GMT-6').dt.tz_convert(LOCAL_TIMEZONE)
    sDf['END_DATE_TIME'] = pd.to_datetime(sDf['END_DATE_TIME'], format = format_string).dt.tz_localize('etc/GMT-6').dt.tz_convert(LOCAL_TIMEZONE)
    sDf = sDf[(sDf['BEGIN_DATE_TIME'] <= end_time) & (sDf['END_DATE_TIME'] >= start_time)]
    return sDf

def get_weather_data(start_time, end_time):
    cid = pd.read_csv(CID_DATA_PATH)
    cid.drop(columns = ['station','dwpc'], inplace = True)
    cid.rename(columns = {
        'valid' : 'time',
        'tmpc' : 't_0m',
        'relh' : 'rh_0m',
        'drct' : 'wd_0m',
        'sped' : 'ws_0m',
        'mslp' : 'p_0m',
        'p01m' : 'precip'
    }, inplace=True)
    cid['time'] = pd.to_datetime(cid['time']).dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
    cid = preprocess.convert_dataframe_units(cid, from_units = CID_UNITS, gravity = LOCAL_GRAVITY, silent = True)
    cid = cid[(cid['time'] <= end_time) & (cid['time'] >= start_time)].reset_index(drop = True)
    return cid

if __name__ == '__main__':
    RELOAD = False
    PARENTDIR = 'C:/Users/22wal/OneDrive/GLWind' # If you are not Elliott and this is not the path for you then pass argument -p followed by the correct path when running!

    if len(sys.argv) > 1:
        if '-r' in sys.argv:
            RELOAD = True
        if '-p' in sys.argv:
            p_index = sys.argv.index('-p')
            if p_index == len(sys.argv) - 1:
                raise('Must follow -p flag with path to parent directory (directory containing /data/), in UNIX format without any quotation marks')
            PARENTDIR = sys.argv[p_index + 1]
    
    if RELOAD:
        df = load_data(
            data_directory = f'{PARENTDIR}/data/KCC_SlowData',
            outer_merges = False,
        ) # Will return with default (not time) index, which is good for the storm/weather data

        print('Getting storm events')
        storm_events = get_storm_events(start_time = START_TIME, end_time = END_TIME, radius = 25, unit = 'km')

        print('Getting weather data')
        cid_data = get_weather_data(start_time = START_TIME, end_time = END_TIME)

        df.set_index('time', inplace=True) # Need time index from here on

        df = perform_preprocessing(
            df = df,
            shadowing_width = 30, # width, in degrees, of shadowing bins
            removal_periods = {
                ('2018-03-05 13:20:00','2018-03-10 00:00:00') : 'ALL', # large maintenance gap
                ('2018-04-18 17:40:00','2018-04-19 14:20:00') : [106], # small maintenance-shaped gap
                ('2018-09-10 12:00:00','2018-09-20 12:00:00') : 'ALL' # blip at end
            },
            outlier_window = 30, # Duration, in minutes, of rolling outlier removal window
            outlier_sigma = 5, # Number of standard deviations beyond which to discard outliers in rolling removal
            resampling_window = 10, # Duration, in minutes, of resampling window
            storm_events = storm_events,
            weather_data = cid_data,
            storm_removal = ['hail','storm','heavy_rain'] # When any of these columns are True, discard the data
        )

        # Define 3-class bulk Richardson number stability classification scheme
        stability_classifier = StabilityClassifier(
            parameter = 'Ri_bulk',
            classes = [
                ('unstable', '(-inf,-0.1)'),
                ('neutral', '[-0.1,0.1]'),
                ('stable', '(0.1,inf)')
            ]
        )

        # Define the terrain classification scheme
        terrain_classifier = TerrainClassifier(
            complexCenter = 315,
            openCenter = 135,
            radius = 30,
            inclusive = True,
            height = 10
        )

        df = compute_values(
            df = df,
            terrain_classifier = terrain_classifier,
            stability_classifier = stability_classifier
        )

        df.reset_index(inplace = True)

        print('Complete, saving results')

        df.to_csv(f'{PARENTDIR}/results/output.csv', index = False)
        storm_events.to_csv(f'{PARENTDIR}/results/storms.csv', index = False)
        cid_data.to_csv(f'{PARENTDIR}/results/cid.csv', index = False)
    else:
        print('RELOAD set to False, will use previous output.')

    print('Loading results')
    df = pd.read_csv(f'{PARENTDIR}/results/output.csv')
    df['time'] = pd.to_datetime(df['time'], utc=True) # will convert to UTC!
    df['time'] = df['time'].dt.tz_convert('US/Central')

    storm_events = pd.read_csv(f'{PARENTDIR}/results/storms.csv')
    
    cid_data = pd.read_csv(f'{PARENTDIR}/results/cid.csv')
    cid_data['time'] = pd.to_datetime(cid_data['time'], utc=True) # will convert to UTC!
    cid_data['time'] = cid_data['time'].dt.tz_convert('US/Central')

    print('Results loaded')

    # print(storm_events[['BEGIN_DATE_TIME','EVENT_TYPE']])
    # print(cid_data)
    # print(df)

    temp_plots(df)
