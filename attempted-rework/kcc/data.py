from windprofiles import MetTower, Boom
import os
import pandas as pd
from definitions import LOCATION, SOURCE_TIMEZONE, SOURCE_UNITS

def _get_boom_objects() -> list[Boom]:
    boom1 = Boom(1, 6., "m",)
    boom2 = Boom(2, 10., "m")
    boom3 = Boom(3, 20., "m")
    boom4 = Boom(4, 32., "m")
    boom5 = Boom(5, 80., "m")
    boom6 = Boom(6, 106., "m")
    boom7 = Boom(7, 106., "m")
    return [boom1, boom2, boom3, boom4, boom5, boom6, boom7]

def _read_csvs_to_dfs(parent) -> list[pd.DataFrame]:
    df1 = pd.read_csv(os.path.join(parent, "Boom1OneMin")).rename(
        columns={
            "TimeStamp": "time",
            "MeanVelocity (m/s)": "ws",
            "MeanDirection": "wd",
            "MeanTemperature (C )": "t",
            "MeanPressure (mmHg)": "p",
        }
    )

    df2 = pd.read_csv(os.path.join(parent, "Boom2OneMin")).rename(
        columns={
            "TIMESTAMP": "time",
            "MeanVelocity (m/s)": "ws_2",
            "MeanDirection": "wd_2",
            "MeanTemperature (C )": "t_2",
            "MeanRH (%)": "rh_2",
        }
    )

    df3 = pd.read_csv(os.path.join(parent, "Boom3OneMin")).rename(
        columns={
            "TIMESTAMP": "time",
            "MeanVelocity (m/s)": "ws_3",
            "MeanDirection": "wd_3",
        }
    )

    df4 = pd.read_csv(os.path.join(parent, "Boom4OneMin")).rename(
        columns={
            "TimeStamp": "time",
            "MeanVelocity": "ws_4",
            "MeanDirection": "wd_4",
            "MeanTemperature": "t_4",
            "MeanRH": "rh_4",
        }
    )

    df5 = pd.read_csv(os.path.join(parent, "Boom5OneMin")).rename(
        columns={
            "TimeStamp": "time",
            "MeanVelocity": "ws_5",
            "MeanDirection": "wd_5",
            "MeanTemperature": "t_5",
            "MeanRH": "rh_5",
        }
    )

    df6 = pd.read_csv(os.path.join(parent, "Boom6OneMin")).rename(
        columns={
            "TIMESTAMP": "time",
            "MeanVelocity (m/s)": "ws_6a",
            "Mean Direction": "wd_6a",
            "MeanTemperature (C )": "t_6",
            "MeanRH (%)": "rh_6",
        }
    )

    df7 = pd.read_csv(os.path.join(parent, "Boom7OneMin")).rename(
        columns={
            "TimeStamp": "time",
            "MeanVelocity (m/s)": "ws_6b",
            "MeanDirection": "wd_6b",
            "MeanPressure (mmHg)": "p_6",
        }
    )

    return [df1, df2, df3, df4, df5, df6, df7]

def load_data(parent: str) -> MetTower:
    booms = _get_boom_objects()
    dfs = _read_csvs_to_dfs(parent)

    for df, boom in zip(booms, dfs):
        boom.add_data(df, units = SOURCE_UNITS, timezone = SOURCE_TIMEZONE)

    tower = MetTower(location = LOCATION, booms = booms)
    return tower
