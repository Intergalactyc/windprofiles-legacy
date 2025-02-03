import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def bar_stability(df):
    pass

def wind_profiles_by_terrain():
    pass

def wse_histograms_by_stability():
    pass

def veer_profiles():
    pass

ALL_PLOTS = {
    'bar_stability': ('Stability Frequency Bar Plot', bar_stability),
    'annual_profiles' : ('Annual Wind Profiles with Fits, by Terrain', wind_profiles_by_terrain),
    'wse_histograms' : ('Histograms of WSE, by Stability, including Terrain', wse_histograms_by_stability),
    'veer_profiles' : ('Wind direction profiles, by Terrain', veer_profiles)
}

def list_possible_plots():
    print('Possible plots to generate:')
    for tag in ALL_PLOTS.keys():
        print(f'\t{tag}')

def generate_plots(df: pd.DataFrame, cid: pd.DataFrame, savedir: str, summary: dict, which: list = ALL_PLOTS.keys()):
    # For now the cid argument is not used but I'm leaving it in case that changes
    print('Generating final plots.')
    not_generated = list(ALL_PLOTS.keys())
    for tag in which:
        long, plotter = ALL_PLOTS[tag]
        print(f"Generating plot {tag}: '{long}'")
        plotter(df = df, summary = summary, saveto = f'{savedir}/{tag}.png')
        not_generated.remove(tag)
    if len(not_generated) != 0:
        print(f'Plots not generated: {not_generated}') 
