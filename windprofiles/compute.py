import pandas as pd
import numpy as np
import windprofiles.lib.atmos as atmos
from windprofiles.classify import TerrainClassifier, PolarClassifier, StabilityClassifier, SingleClassifier
from warnings import warn

STANDARD_GRAVITY = 9.80665 # standard gravitational parameter g in m/s^2

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

def classifications(df: pd.DataFrame, *, terrain_classifier: PolarClassifier|TerrainClassifier = None, stability_classifier: SingleClassifier|StabilityClassifier = None) -> pd.DataFrame:
    """
    Classify terrain and/or stability for each timestamp in a dataframe.
    Creates a new column in the dataframe for each type of result.
    """
    if terrain_classifier is None and stability_classifier is None:
        warn('Neither terrain nor stability classifier passed')
        
    result = df.copy()

    if terrain_classifier is not None:
        result['terrain'] = terrain_classifier.classify_rows(result)
    if stability_classifier is not None:
        result['stability'] = stability_classifier.classify_rows(result)

    return result
