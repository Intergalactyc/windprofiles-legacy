import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import os

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
