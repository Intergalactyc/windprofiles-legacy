import pandas as pd
import numpy as np
import windprofiles.preprocess as preprocess
import windprofiles.compute as compute
from windprofiles.classify import TerrainClassifier, StabilityClassifier
from windprofiles.analyze import save
import sys
import os

# Timezones (source: data, local: local timezone of collection location)
SOURCE_TIMEZONE = 'UTC'
LOCAL_TIMEZONE = 'US/Central'

# Local gravity at Cedar Rapids (latitude ~ 42 degrees, elevation ~ 247 m), in m/s^2
LOCAL_GRAVITY = 9.802

# Latitude and longitude of KCC met tower, each in degrees
LATITUDE = 41.91
LONGITUDE = -91.65

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

    return df
    
def perform_preprocessing(df,
                          shadowing_width,
                          removal_periods, # These will be passed in UTC so we will do the removal before converting data time to local US/Central
                          outlier_window,
                          outlier_sigma,
                          resampling_window):
    print("START DATA PREPROCESSING")
    # Automatically convert units of all columns to standards as outlined in convert.py
    df = preprocess.convert_dataframe_units(df = df, from_units = SOURCE_UNITS)

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

    # Remove rows where there isn't enough data
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
    import ipplot as plot
    #plot.hist_alpha_by_stability(df, separate = True, compute = True, overlay = True)
    #plot.alpha_tod_violins(df, fit = False)

    tod_dir = 'C:/Users/22wal/OneDrive/Pictures/temp/tods_wider'
    plot.alpha_tod_violins(df, fit = True, saveto = f'{tod_dir}/year.png')
    plot.alpha_tod_violins_by_terrain(df, fit = True, saveto = f'{tod_dir}/yearT.png')
    for season in ['Fall', 'Winter', 'Spring', 'Summer']:
        plot.alpha_tod_violins(df, season = season, fit = True, saveto = f'{tod_dir}/{season.lower()}.png')
        plot.alpha_tod_violins_by_terrain(df, season = season, fit = True, saveto = f'{tod_dir}/{season.lower()}T.png') 

    #plot.ri_tod_violins(df, fit = False, cut = 25, printcutfrac = True, bounds = (-5,3))
    ###plot.boom_data_available(df, freq = '10min', heights = HEIGHTS)
    #plot.alpha_over_time(df)
    #plot.comparison(df, which = ['alpha', 'alpha_no106'], xlims=(-0.5,1), ylims = (-0.5,1))
    #plot.comparison(df, which = ['alpha', 'alpha_no32'], xlims=(-0.5,1), ylims = (-0.5,1))

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
        )

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

        save(df, f'{PARENTDIR}/results/output.csv')
    else:
        print('RELOAD set to False, will use previous output.')

    df = pd.read_csv(f'{PARENTDIR}/results/output.csv')
    df['time'] = pd.to_datetime(df['time'], utc=True) # will convert to UTC!
    df['time'] = df['time'].dt.tz_convert('US/Central')
    
    temp_plots(df)

    print(len(df[df['alpha'] < 0]))
    print(len(df))
    