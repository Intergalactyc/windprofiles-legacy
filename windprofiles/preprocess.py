# preprocess.py
# Unit conversion, merging booms at a common height, initial QC

import numpy as np
import pandas as pd
import windprofiles.lib.polar as polar
import windprofiles.lib.atmos as atmos
import warnings
from datetime import timedelta

# DO NOT CHANGE STANDARDS WITHOUT ALL CORRESPONDING UPDATES
_standards = {
    'p' : 'kPa', # pressure
    't' : 'K', # temperature
    'rh' : 'decimal', # relative humidity
    'ws' : 'm/s', # wind speed
    'wd' : ('degrees', 'N', 'CW'), # wind direction [angle measure, zero point, orientation]
}

def get_standards():
    return _standards

def print_standards():
    print(_standards)

def _convert_pressure(series, from_unit, gravity = atmos.STANDARD_GRAVITY):
    """
    Conversion of pressure units
    If input has format "{unit}_{number}asl", interpreted
        as sea-level pressure and converted to pressure
        at height of <number> meters
    """
    if _standards['p'] != 'kPa':
        raise(f'preprocess._convert_pressure: Standardized pressure changed from kPa to {_standards["p"]} unexpectedly')
    if '_' in from_unit:
        from_unit, masl = from_unit.split('_')
        meters_asl = float(masl[:-3])
        series = atmos.pressure_above_msl(series, meters_asl, gravity = gravity)
    match from_unit:
        case 'kPa':
            return series
        case 'mmHg':
            return series * 0.13332239
        case 'mBar':
            return series / 10.
        case _:
            raise(f'preprocess._convert_pressure: Unrecognized pressure unit {from_unit}')
        
def _convert_temperature(series, from_unit):
    """
    Conversion of temperature units
    """
    if _standards['t'] != 'K':
        raise(f'preprocess._convert_temperature: Standardized temperature changed from K to {_standards["t"]} unexpectedly')
    match from_unit:
        case 'K':
            return series
        case 'C':
            return series + 273.15
        case 'F':
            return (series - 32) * (5/9) + 273.15
        case _:
            raise(f'preprocess._convert_temperature: Unrecognized temperature unit {from_unit}')

def _convert_humidity(series, from_unit):
    """
    Relative humidity conversions.
    Does not account for other types of humidity - 
        just for switching between percent [%] (0-100)
        and decimal (0-1) scales
    """
    if _standards['rh'] != 'decimal':
        raise(f'preprocess._convert_humidity: Standardized relative humidity changed from decimal to {_standards["rh"]} unexpectedly')
    match from_unit:
        case 'decimal':
            return series
        case '.':
            return series
        case '%':
            return series / 100.
        case 'percent':
            return series / 100.
        case _:
            raise(f'preprocess._convert_humidity: Unrecognized humidity unit {from_unit}')

def _convert_speed(series, from_unit):
    """
    Conversion of wind speed units
    """
    if _standards['ws'] != 'm/s':
        raise(f'preprocess._convert_speed: Standardized wind speed changed from m/s to {_standards["ws"]} unexpectedly')
    match from_unit:
        case 'm/s':
            return series
        case 'mph':
            return series / 2.23694
        case 'mi/hr':
            return series / 2.23694
        case 'mi/h':
            return series / 2.23694
        case _:
            raise(f'preprocess._convert_speed: Unrecognized wind speed unit {from_unit}')

def _convert_direction(series, from_unit):
    """
    Conversion of wind direction
    """
    if _standards['wd'] != ('degrees', 'N', 'CW'):
        raise(f'preprocess._convert_direction: Standardized wind speed changed from degrees CW of N to {_standards["wd"][0]} {_standards["wd"][2]} of {_standards["wd"][1]} unexpectedly')
    measure, zero, orient = from_unit
    
    # Convert measure to degrees (possibly from radians)
    if measure in ['rad', 'radians']:
        series = np.rad2deg(series)
    elif measure not in ['deg', 'degrees']:
        raise(f'preprocess._convert_direction: Unrecognized angle measure {measure}')
    
    # Convert orientation to clockwise (possibly from counterclockwise)
    if orient.lower() in ['ccw', 'counterclockwise']:
        series = (-series) % 360
    elif orient.lower() not in ['cw', 'clockwise']:
        raise(f'preprocess._convert_direction: Unrecognized angle orientation {orient}')
    
    # Align zero point to north
    if type(zero) is str:
        # From cardinal direction
        match zero.lower():
            case 'n':
                return series
            case 'e':
                return (series - 90) % 360
            case 's':
                return (series - 180) % 360
            case 'w':
                return (series - 270) % 360
    elif type(zero) in [int, float]:
        # From degrees offset
        return (series - zero) % 360
    else:
        raise(f'preprocess._convert_direction: Unrecognized zero type {type(zero)} for {zero}')

def convert_dataframe_units(df, from_units, gravity = atmos.STANDARD_GRAVITY, silent = False):
    """
    Public function for converting units for all 
    (commonly formatted) columns in dataframe based
    on a dictionary of units, formatted in the same
    way as the _standards (use get_standards() or
    print_standards() to view)
    """

    result = df.copy(deep = True)

    conversions_by_type = {
        'p' : _convert_pressure,
        't' : _convert_temperature,
        'rh' : _convert_humidity,
        'ws' : _convert_speed,
        'wd' : _convert_direction,
    }

    for column in result.columns:
        if '_' in column and 'time' not in column:
            column_type = column.split('_')[0]
            if column_type in conversions_by_type.keys():
                conversion = conversions_by_type[column_type]
                if column_type == 'p':
                    result[column] = conversion(series = result[column],
                                                from_unit = from_units[column_type],
                                                gravity = gravity)
                else:
                    result[column] = conversion(series = result[column],
                                                from_unit = from_units[column_type])

    if not silent:
        print('preprocess.convert_dataframe_units() - DataFrame unit conversion completed')

    return result

def clean_formatting(df, type = 'float32', silent = False):
    """
    At times when wind speed for a certain height
        is zero, sets the corresponding wind direction
        to NaN (np.nan).
    Also cast data (with '_', o.t. times) to `type`, default float32.
        Disable this by setting `type = None`.
    Finally, fixes duplicates and misordering.
    Assumes that dataframe formatting is already
        otherwise full consistent with guidelines.
    """
    result = df.copy(deep = True)

    for column in result.columns:
        if '_' in column and 'time' not in column:
            result[column] = result[column].astype(type)
            columntype, heightStr, *_ = column.split('_')
            if columntype == 'ws':
                dircol = f'wd_{heightStr}'
                result.loc[result[column] == 0, dircol] = np.nan

    result = result.reset_index(names = 'time').sort_values(by = 'time').set_index('time')
    result = result[~result.index.duplicated(keep = 'first')]
    
    if not silent:
        print('preprocess.clean_formatting() - completed formatting update')
    
    return result

def shadowing_merge(df,
                    speeds,
                    directions,
                    angles,
                    width = 30,
                    drop_old = True,
                    silent = False):
    """
    Merges multiple sets of data at a shared height, accounting
        for tower shadowing effects.
    `speeds` and `directions` should be the names of the columns of
        `df` containing the wind speeds and directions to combine.
    `angles` should be the center wind direction from which 
        shadowing occurs for their respective boom (data set).
    `speeds`, `directions`, and `angles` must be iterables 
        of the same length. 
    At each time, if the wind direction reported by boom `i` is
        within width/2 of its corresponding shadowing angle,
        then its data will be considered shadowed and neither its
        speed or direction will be used. Data from all booms which
        are not shadowed will be (vector) averaged to form the
        resulting wind speed and direction at that time.
    Returns two columns: merged wind speeds and merged wind directions.
    """
    if not (len(speeds) == len(directions) == len(angles)):
        raise(f'preprocess.shadowing_merge: Mismatched lengths for speeds/directions/angles (given lengths {len(speeds)}/{len(directions)}/{len(angles)})')
    nBooms = len(speeds)
    radius = width / 2
    raw_deviations = [(df[dir] - ang) % 360 for dir, ang in zip(directions, angles)]
    indexer = [col.apply(lambda d : min(360 - d, d) > radius) for col in raw_deviations]
    n_shadowed = [len(indexer[i]) - indexer[i].sum() for i in range(nBooms)]
    uList = []
    vList = []
    for i in range(nBooms):
        _spd, _dir, _ang = speeds[i], directions[i], angles[i]
        raw_deviations = (df[_dir] - _ang) % 360
        corr_deviations = raw_deviations.apply(lambda d : min(360 - d, d))
        u, v = polar.wind_components(df[_spd], df[_dir])
        u.loc[corr_deviations < radius] = np.nan
        v.loc[corr_deviations < radius] = np.nan
        uList.append(u)
        vList.append(v)
    # We want the mean(np.nan...) -> np.nan behavior and expect to see it sometimes, so we'll filter the error
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    uMeans = np.nanmean(np.stack(uList), axis = 0)
    vMeans = np.nanmean(np.stack(vList), axis = 0)
    sMeans = np.sqrt(uMeans * uMeans + vMeans * vMeans)
    dMeans = (np.rad2deg(np.arctan2(uMeans, vMeans)) + 360) % 360
    if drop_old:
        df.drop(columns = speeds + directions, inplace = True)
    if not silent:
        print('preprocess.shadowing_merge() - completed merge')
        print(f'\tNumber of data points with shadowing, by boom: {[int(n) for n in n_shadowed]}')
    return sMeans, dMeans

def remove_data(df: pd.DataFrame, periods: dict, silent: bool = False) -> pd.DataFrame:
    """
    Removes data within certain specified datetime intervals.
    Removal can be complete (specify 'ALL') or partial (specify 
        list of integer heights).
    See kcc.py's `removal_periods` for an example of proper
        format for `periods`.
    If silent == False then #s of total and partial removals
        will be printed.
    """
    result = df.reset_index(names = 'time')
    
    total_removals = 0
    partial_removals = 0
    
    for interval, which_heights in periods.items():
        
        removal_start, removal_end = interval
        indices = result.loc[result['time'].between(removal_start, removal_end, inclusive='both')].index
        
        if type(which_heights) is str and which_heights.lower() == 'all': # if all data is to be removed, just drop the full row entry
            result.drop(index = indices, inplace = True)
            total_removals += len(indices)
        elif type(which_heights) is list: # otherwise, just set data from the selected heights to NaN
            datatypes = ['p','ws','wd','t','rh']
            for h in which_heights:
                for d in datatypes:
                    selection = f'{d}_{h}m'
                    if selection in result.columns:
                        result.loc[indices, selection] = np.nan
            partial_removals += len(indices)
        else:
            raise('preprocess.remove_data: Unrecognized removal-height specification in given periods', periods)
    
    result.set_index('time', inplace = True)
    
    if not silent:
        print('preprocess.remove_data() - completed interval data removal')
        print(f'\tTotal removals: {total_removals}')
        print(f'\tPartial removals: {partial_removals}')
    
    return result

def rolling_outlier_removal(df: pd.DataFrame,
                            window_size_minutes: int = 30,
                            sigma: int = 5,
                            column_types = ['ws', 't', 'p', 'rh'],
                            silent: bool = False) -> pd.DataFrame:
    """
    Eliminate data where values from columns of types `column_types` are more than
        `sigma` (default 5) standard deviations from a rolling mean, rolling in a
        window of width `window_size_minutes` (default 30) minutes. 
    Unable to handle wind direction - don't try to apply it to 'wd'.
    """
    result = df.copy(deep = True)
    window = f'{window_size_minutes}min'
    eliminations = 0

    for column in result.columns:
        column_type = column.split('_')[0]
        if column_type in column_types:
            rolling_mean = result[column].rolling(window = window).mean()
            rolling_std = result[column].rolling(window = window).std()
            threshold = sigma * rolling_std
            outliers = np.abs(result[column] - rolling_mean) > threshold
            eliminations += result[outliers].shape[0]
            result = result[~outliers]
    
    if not silent:
        print('preprocess.rolling_outlier_removal() - outlier removal complete')
        print(f'\t{eliminations} outliers eliminated ({100*eliminations/(df.shape[0]):.4f}%)')
    
    return result

def resample(df: pd.DataFrame,
             all_heights: list[int],
             window_size_minutes: int,
             how: str = 'mean',
             silent: bool = False) -> pd.DataFrame:
    
    to_resample = df.copy(deep = True)
    easy_cols = ['t', 'p', 'rh']
    window = f'{window_size_minutes}min'

    for h in all_heights:
        dirRad = np.deg2rad(to_resample[f'wd_{h}m'])
        to_resample[f'x_{h}m'], to_resample[f'y_{h}m'] = polar.wind_components(to_resample[f'ws_{h}m'], to_resample[f'wd_{h}m'])
    
    if how == 'mean':
        resampled = to_resample.resample(window).mean()
    elif how == 'median':
        resampled = to_resample.resample(window).median()
    else:
        raise(f'preprocess.resample: Unrecognized resampling method {how}')
    
    before_drop = resampled.shape[0]
    resampled.dropna(axis = 0, how = 'all', inplace = True)
    dropped = before_drop - resampled.shape[0]

    for h in all_heights:
        resampled[f'ws_{h}m'] = np.sqrt(resampled[f'x_{h}m']**2+resampled[f'y_{h}m']**2)
        resampled[f'wd_{h}m'] = (np.rad2deg(np.arctan2(resampled[f'x_{h}m'], resampled[f'y_{h}m'])) + 360) % 360
        resampled.drop(columns=[f'x_{h}m',f'y_{h}m'], inplace=True)

    if not silent:
        print(f'preprocess.resample() - resampling into {window_size_minutes} minute intervals ({how}s) complete')
        print(f'\tSize reduced from {to_resample.shape[0]} to {before_drop}, before removals')
        print(f'\t{dropped} removals of NaN rows ({100*dropped/before_drop:.4f}%), resulting in {resampled.shape[0]} final data points')
    
    return resampled

def convert_timezone(df: pd.DataFrame, source_timezone: str, target_timezone: str):
    result = df.copy()
    result.index = df.index.tz_localize(source_timezone).tz_convert(target_timezone)
    return result

def determine_weather(df: pd.DataFrame, storm_events: pd.DataFrame, weather_data: pd.DataFrame, trace_float: float = 0.) -> pd.DataFrame:
    HOUR = timedelta(hours = 1)
    # mark times inclusively between storm event start and end times as either hail = True or storm = True
    result = df.copy()
    all_storms = list(storm_events.apply(lambda row : (row['BEGIN_DATE_TIME'], row['END_DATE_TIME'], row['EVENT_TYPE']), axis = 1))
    result['hail'] = False
    result['storm'] = False
    result['heavy_rain'] = False
    result['light_rain'] = False
    for start, end, storm_type in all_storms:
        if start == end:
            start -= 1.5 * HOUR
            end += 1.5 * HOUR
        if storm_type == 'Hail':
            result.loc[(result.index >= start) & (result.index <= end), 'hail'] = True
        elif storm_type.lower() == 'Flash Flood':
            result.loc[(result.index >= start) & (result.index <= end), 'heavy_rain'] = True
        else:
            result.loc[(result.index >= start) & (result.index <= end), 'storm'] = True
    # mark times where precipitation is above trace value as rain = True
        # for each time stamp in the CID data where it is raining, mark df time stamps between the previous timestamp and now as raining
    for index, row in weather_data.iterrows(): # I know this is slower than optimal but there isn't THAT much data and it works fine
        precip = row['precip']
        if precip > trace_float and index > 0: # not really handling the case where index is 0 but we know it's not raining at the start anyway
            start = row['time'] - HOUR
            end = row['time']
            selector = 'light_rain' if precip < 5 else 'heavy_rain'
            result.loc[(result.index >= start) & (result.index <= end), selector] = True
    return result

def flagged_removal(df: pd.DataFrame, flags: str|list[str], silent: bool = False, drop_cols = True):
    """
    For each column listed in `flags`, remove rows from `df` where that column is True
    """
    if not silent:
        print(f'preprocess.flagged_removal() - beginning removals based on column(s) {flags}')

    original_size = len(df)
    result = df.copy()

    if type(flags) is str:
        flags = [flags]

    for flag in flags:
        print(result[flag])
        result.drop(result[result[flag] == True].index, inplace = True)

    result.drop(columns = flags, inplace = True)

    if not silent:
        removals = original_size - len(result)
        print(f'\tRemovals complete ({removals} rows dropped, {len(result)} rows remain)')

    return result

def strip_missing_data(df: pd.DataFrame, necessary: list[int], minimum: int = 4, silent: bool = False):
    """
    Remove rows where there are fewer than `minimum` wind speed columns or where
        wind speeds are missing at any of the `necessary` heights
    """
    result = df.copy()

    if not silent:
        print('preprocess.strip_missing_data() - beginning removals')

    cols = result.columns

    necessarys = [f'ws_{h}m' for h in necessary]
    ws_cols = []
    for col in cols:
        if 'ws_' in col:
           ws_cols.append(col) 

    removed = 0
    iterable = result.iterrows()
    for index, row in iterable:
        drop = False
        for necessary in necessarys:
            if pd.isna(row[necessary]):
                drop = True
                break
        count = 0
        for col in ws_cols:
            if not pd.isna(row[col]):
                count += 1
        if drop or count < minimum:
            result.drop(index = index, inplace = True)
            removed += 1

    if not silent:
        print(f'\tRemovals complete ({removed} rows dropped)')

    return result
