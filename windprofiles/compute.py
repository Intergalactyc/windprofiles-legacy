import pandas as pd
import numpy as np
import windprofiles.lib.atmos as atmos
import windprofiles.lib.stats as stats
from windprofiles.classify import TerrainClassifier, PolarClassifier, StabilityClassifier, SingleClassifier
from warnings import warn

STANDARD_GRAVITY = 9.80665 # standard gravitational parameter g in m/s^2

def strip_failures(df: pd.DataFrame, subset: list[str], silent: bool = False):
    if not silent:
        print('compute.strip_failures() - removing rows where necessary computations failed')

    result = df.dropna(axis = 'rows', how = 'any', subset = subset)

    if not silent:
        n_dropped = len(df) - len(result)
        print(f'\tRemoved {n_dropped} rows, {len(result)} remain')

    return result

def virtual_potential_temperatures(df: pd.DataFrame, heights: list[int], *, silent: bool = False, substitutions: dict[str:str] = None) -> pd.DataFrame:
    """
    Compute virtual potential temperatures at all given heights.
    Creates new columns in the dataframe with the results.
    """
    result = df.copy()

    if not silent:
        print('compute.virtual_potential_temperatures() - computing virtual potential temperatures')

    for h in heights:

        rh_str = f'rh_{h}m'
        p_str = f'p_{h}m'
        t_str = f't_{h}m'

        if rh_str in substitutions.keys():
            rh_str = substitutions[rh_str]
        if p_str in substitutions.keys():
            p_str = substitutions[p_str]
        if t_str in substitutions.keys():
            t_str = substitutions[t_str]

        result[f'vpt_{h}m'] = atmos.vpt_from_3(relative_humidity = result[rh_str],
                         barometric_air_pressure = result[p_str],
                         temperature = result[t_str])
        
        if not silent:
            print(f'\tCompleted computation at height {h}m')

    return result

def environmental_lapse_rate(df: pd.DataFrame, variable: str, heights: list[int, int], *, silent: bool = False) -> pd.DataFrame:
    """
    Approximate environmental lapse rate of a variable between two heights.
    Creates a new column in the dataframe with the results.
    """
    if not silent:
        print(f'compute.environmental_lapse_rate() - computing lapse rate of {variable}')

    if type(heights) not in [list, tuple] or len(heights) != 2 or heights[0] == heights[1]:
        raise(f'compute.environmental_lapse_rate: invalid heights {heights}')
    if type(variable) is not str:
        raise(f'compute.environmental_lapse_rate: invalid variable {variable}')

    h1 = int(min(heights))
    h2 = int(max(heights))
    h1_str = f'{variable}_{h1}m'
    h2_str = f'{variable}_{h2}m'

    result = df.copy()

    if not h1_str in result.columns:
        raise(f'compute.environmental_lapse_rate: {h1_str} not found in DataFrame columns')
    if not h2_str in result.columns:
        raise(f'compute.environmental_lapse_rate: {h2_str} not found in DataFrame columns')
    
    result[f'{variable}_lapse'] = (result[h2_str] - result[h1_str])/(h2 - h1)

    if not silent:
        print(f'\tCompleted computation between heights {h1} and {h2}')

    return result

def bulk_richardson_number(df: pd.DataFrame, heights: list[int, int], *, silent: bool = False, gravity: float = STANDARD_GRAVITY) -> pd.DataFrame:
    """
    Compute bulk Richardson number Ri_bulk using data at two heights.
    Creates a new column in the dataframe with the results. 
    """
    if not silent:
        print(f'compute.bulk_richardson_number() - computing bulk Ri')

    if type(heights) not in [list, tuple] or len(heights) != 2 or heights[0] == heights[1]:
        raise(f'compute.environmental_lapse_rate: invalid heights {heights}')
    
    h_lower = int(min(heights))
    h_upper = int(max(heights))

    result = df.copy()
    result['Ri_bulk'] = result.apply(lambda row : atmos.bulk_richardson_number(row[f'vpt_{h_lower}m'], row[f'vpt_{h_upper}m'], h_lower, h_upper, row[f'ws_{h_lower}m'], row[f'ws_{h_upper}m'], row[f'wd_{h_lower}m'], row[f'wd_{h_upper}m'], gravity = gravity), axis = 1)

    if not silent:
        print(f'\tCompleted computation between heights {h_lower} and {h_upper}')

    return result

def classifications(df: pd.DataFrame, *, terrain_classifier: PolarClassifier|TerrainClassifier = None, stability_classifier: SingleClassifier|StabilityClassifier = None, silent: bool = False) -> pd.DataFrame:
    """
    Classify terrain and/or stability for each timestamp in a dataframe.
    Creates a new column in the dataframe for each type of result.
    """
    if terrain_classifier is None and stability_classifier is None:
        warn('Neither terrain nor stability classifier passed')

    if not silent:
        tc = terrain_classifier is not None
        sc = stability_classifier is not None
        bth = tc and sc
        print(f'compute.classifications() - classifying {"terrain" * tc}{" and " * bth}{"stability" * sc}')
        
    result = df.copy()

    if terrain_classifier is not None:
        result['terrain'] = terrain_classifier.classify_rows(result)
    if stability_classifier is not None:
        result['stability'] = stability_classifier.classify_rows(result)

    if not silent:
        print('\tCompleted classifications')

    return result

def power_law_fits(df: pd.DataFrame, heights: list[int], minimum_present: int = 2, columns: list[str, str] = ['beta', 'alpha'], silent: bool = False):
    """
    Fit power law u(z) = A z ^ B to each timestamp in a dataframe.
    Creates new columns columns[0] and column[1] for the coefficients
        A and B ('beta' and 'alpha' by default) respectively.
    """
    if not silent:
        print(f'compute.power_law_fits() - computing power law fits using {heights}')

    if type(minimum_present) is not int or minimum_present < 2:
        raise(f'windprofiles.compute.power_law_fits - invalid argument \'{minimum_present}\' passed to \'minimum_present\'')
    if len(heights) < minimum_present:
        raise(f'windprofiles.compute.power_law_fits - insufficient number of heights provided') 
    if type(columns) not in [tuple, list] or len(columns) != 2:
        raise(f'windprofiles.compute.power_law_fits - \'columns\' must be a tuple or list of two column names for the multiplicative coefficient and power, respectively')

    result = df.copy()

    result[['A_PRIMITIVE', 'B_PRIMITIVE']] = result.apply(lambda row : stats.power_fit(heights, [row[f'ws_{height}m'] for height in heights], require = minimum_present), axis = 1, result_type='expand')

    if columns[0] is None:
        result.drop(columns = ['A_PRIMITIVE'], inplace = True)
    elif type(columns[0]) is str:
        result.rename(columns = {'A_PRIMITIVE' : columns[0]}, inplace = True)
    if columns[1] is None:
        result.drop(columns = ['B_PRIMITIVE'], inplace = True)
    elif type(columns[1]) is str:
        result.rename(columns = {'B_PRIMITIVE' : columns[1]}, inplace = True)

    if not silent:
        print(f'\tCompleted computation, multiplicative coefficient stored in {columns[0]} and exponent stored in {columns[1]}')

    return result
