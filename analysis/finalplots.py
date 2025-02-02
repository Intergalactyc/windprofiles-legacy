import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

COLORS = {
    'open' : '#1f77b4', # tab:blue C0
    'complex' : '#ff7f0e', # tab:orange C1
    'unstable' : '#ef476f', # bright pink
    'neutral' : '#ffd166', # sunglow yellow (or try #ffa62b)
    'stable' : '#06d6a0', # emerald green
    'strongly stable' : '#17becf', # robin egg blue
    'default1' : '#7f7f7f', # C7 gray
    'default2' : '#2ca02c', # C2 green
}

def bar_stability(df, summary, size, saveto, suptitle):
    if summary['stability_classes'] == 4:
        fig, ax = plt.subplots(figsize = size)
        stability_r_freqs = 100 * df['stability'].value_counts(normalize = True)
        ax.bar(
            x = ['Unstable\n(Ri<-0.1)','Neutral\n(-0.1<Ri<0.1)','Stable\n(0.1<Ri<0.25)','Strongly Stable\n(0.25<Ri)'],
            height = [stability_r_freqs['unstable'], stability_r_freqs['neutral'], stability_r_freqs['stable'], stability_r_freqs['strongly stable']],
            color = [COLORS['unstable'], COLORS['neutral'], COLORS['stable'], COLORS['strongly stable']]
        )
        if suptitle:
            ax.set_title('Frequency of Wind Data Sorted by \nBulk Richardson Number Thermal Stability Classification')
        ax.set_ylabel('Proportion of Data (%)')
        ax.grid(axis = 'y', linestyle = '-', alpha = 0.8)
        plt.savefig(saveto, bbox_inches = 'tight')
    else:
        print(f"Cannot handle {summary['stability_classes']} stability classes in plot 'bar_stability'")
    return

def wind_profiles_by_terrain(df, summary, size, saveto, suptitle):
    pass

def wse_histograms_by_stability(df, summary, size, saveto, suptitle):
    pass

def veer_profiles(df, summary, size, saveto, suptitle):
    pass

def tod_wse_plots(df, summary, size, saveto, suptitle):
    pass

def data_gaps_visualization(df, summary, size, saveto, suptitle):
    pass

def terrain_breakdown(df, summary, size, saveto, suptitle):
    pass

ALL_PLOTS = {
    'bar_stability': ('Stability Frequency Bar Plot', bar_stability, (8,6)),
    'annual_profiles' : ('Annual Wind Profiles with Fits, by Terrain', wind_profiles_by_terrain, (10,6)),
    'wse_histograms' : ('Histograms of WSE, by Stability, including Terrain', wse_histograms_by_stability, (10,6)),
    'veer_profiles' : ('Wind direction profiles, by Terrain', veer_profiles, (10,6))
}

def list_possible_plots():
    print('Possible plots to generate:')
    for tag in ALL_PLOTS.keys():
        print(f'\t{tag}')

def generate_plots(df: pd.DataFrame, cid: pd.DataFrame, savedir: str, summary: dict, which: list = ALL_PLOTS.keys(), suptitles: bool = False, fontsize: int = 14):
    plt.rcParams['font.size'] = fontsize
    # For now the cid argument is not used but I'm leaving it in case that changes
    print(f'Generating final plots. Titles are {"on" if suptitles else "off"}, and fontsize = {fontsize}.')
    not_generated = list(ALL_PLOTS.keys())
    for tag in which:
        long, plotter, size = ALL_PLOTS[tag]
        print(f"Generating plot {tag}: '{long}'")
        plotter(df = df, summary = summary, size = size, saveto = f'{savedir}/{tag}.png', suptitle = suptitles)
        not_generated.remove(tag)
    if len(not_generated) != 0:
        print(f'Plots not generated: {not_generated}') 
    print('Rules used in analysis to create plot data:')
    for key, val in summary.items():
        if key[0] != '_':
            print(f'\t{key} = {val}')
