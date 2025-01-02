import numpy as np
import pandas as pd

def wind_components(speed, direction, invert = False):
    # given a wind speed and a direction in degrees CW of N,
    # return u, v (eastward, northward) components of wind
    # if type(speed) is pd.DataFrame:
    #     indexer = math.isnan(direction) | speed == 0.
    #     speed.loc[indexer] = 
    #     speed.loc[speed == 0.]
    # elif math.isnan(direction) or speed == 0.:
    #     return 0., 0.
    direction_rad = np.deg2rad(direction)
    u = speed * np.sin(direction_rad)
    v = speed * np.cos(direction_rad)
    if invert:
        u *= -1
        v *= -1
    u.loc[speed == 0] = 0.
    v.loc[speed == 0] = 0.
    return u, v
