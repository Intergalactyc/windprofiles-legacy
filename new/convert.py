# convert.py
# Unit conversion and merging booms at a common height

import numpy as np
import pandas as pd
import lib.polar as polar
import warnings

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

def _convert_pressure(series, from_unit):
    """
    Conversion of pressure units
    """
    if _standards['p'] != 'kPa':
        raise('Standardized pressure changed from kPa')
    match from_unit:
        case 'kPa':
            return series
        case 'mmHg':
            return series * 0.13332239
        case _:
            raise(f'Unrecognized pressure unit {from_unit}')
        
def _convert_temperature(series, from_unit):
    """
    Conversion of temperature units
    """
    if _standards['t'] != 'K':
        raise('Standardized temperature changed from K')
    match from_unit:
        case 'K':
            return series
        case 'C':
            return series + 273.15
        case 'F':
            return (series - 32) * (5/9) + 273.15
        case _:
            raise(f'Unrecognized temperature unit {from_unit}')

def _convert_humidity(series, from_unit):
    """
    Relative humidity conversions.
    Does not account for other types of humidity - 
        just for switching between percent [%] (0-100)
        and decimal (0-1) scales
    """
    if _standards['rh'] != 'decimal':
        raise('Standardized humidity changed from decimal')
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
            raise(f'Unrecognized humidity unit {from_unit}')

def _convert_speed(series, from_unit):
    """
    Conversion of wind speed units
    """
    if _standards['ws'] != 'm/s':
        raise('Standardized wind speed changed from m/s')
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
            raise(f'Unrecognized wind speed unit {from_unit}')

def _convert_direction(series, from_unit):
    """
    Conversion of wind direction
    """
    if _standards['wd'] != ('degrees', 'N', 'CW'):
        raise('Standardized wind direction configuration changed')
    measure, zero, orient = from_unit
    # Convert measure to degrees (possibly from radians)
    if measure in ['rad', 'radians']:
        series = np.rad2deg(series)
    elif measure not in ['deg', 'degrees']:
        raise(f'Unrecognized angle measure {measure}')
    # Convert orientation to clockwise (possibly from counterclockwise)
    if orient.lower() in ['ccw', 'counterclockwise']:
        series = (-series) % 360
    elif orient.lower() not in ['cw', 'clockwise']:
        raise(f'Unrecognized angle orientation {orient}')
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
        raise(f'Unrecognized zero type {type(zero)} for {zero}')

def convert_dataframe_units(df, from_units):
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
        columntype = column.split('_')[0]
        if columntype in conversions_by_type.keys():
            conversion = conversions_by_type[columntype]
            result[column] = conversion(series = result[column],
                                        from_unit = from_units[columntype])

    return result

def clean_formatting(df, type = 'float32'):
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
    result = result.reset_index().sort_values(by = 'time').set_index('time')
    result = result[~result.index.duplicated(keep = 'first')]
    return result

def shadowing_merge(df,
                    speeds,
                    directions,
                    angles,
                    width = 30,
                    drop_old = True):
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
        raise('Mismatched lengths for speeds/directions/angles')
    nBooms = len(speeds)
    radius = width / 2
    raw_deviations = [(df[dir] - ang) % 360 for dir, ang in zip(directions, angles)]
    indexer = [col.apply(lambda d : min(360 - d, d) > radius) for col in raw_deviations]
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
    return sMeans, dMeans
