import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import os
from functools import reduce

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def get_monthly_breakdown(df: pd.DataFrame, column: str, ignore: list = []) -> tuple[pd.DataFrame]:
    """
    Given a dataframe df with a datetime column 'time' as well as the name of a column
        of interest, returns the breakdown of the amount of entries with each value
        in that column, both as number (breakdown, first return value) and fraction of total
        (proportions, second return value)
    """
    classes = [cls for cls in df[column].unique() if cls not in ignore]
    breakdown = pd.DataFrame(index = MONTHS, columns = classes)
    proportions = breakdown.copy()
    for i, mon in enumerate(MONTHS, 1):
        df_mon = df[df['time'].dt.month == i]
        total = len(df_mon)
        for cls in classes:
            df_cls = df_mon[df_mon[column] == cls]
            count = len(df_cls)
            breakdown.loc[mon, cls] = count
            proportions.loc[mon, cls] = count / total
    return breakdown, proportions

def dict_checksum(d: dict, verbose: bool = False) -> int:
    result = abs(reduce(lambda x,y : x^y, [hash(item) for item in d.items()]))
    if verbose:
        print(result)
    return result

def dataframe_checksum(df: pd.DataFrame, verbose: bool = False) -> int:
    result = int(pd.util.hash_pandas_object(df).sum())
    if verbose:
        print(result)
    return result
