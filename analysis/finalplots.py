import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import windprofiles.lib.stats as stats
import windprofiles.lib.polar as polar

COLORS_POSTER = {
    'open' : '#1f77b4', # tab:blue C0
    'complex' : '#ff7f0e', # tab:orange C1
    'unstable' : '#ef476f', # bright pink
    'neutral' : '#ffd166', # sunglow yellow (or try #ffa62b)
    'stable' : '#06d6a0', # emerald green
    'strongly stable' : '#17becf', # robin egg blue
    'default1' : '#7f7f7f', # C7 gray
    'default2' : '#2ca02c', # C2 green
}

COLORS_FORMAL = {
    'open' : 'tab:blue',
    'complex' : 'tab:orange',
    'unstable' : 'tab:red',
    'neutral' : '#3b50d6',
    'stable' : '#9b5445',
    'strongly stable' : 'tab:green',
    'default1' : 'tab:blue',
    'default2' : 'tab:orange'
}

MARKERS = {
    'open' : 'o',
    'complex' : 'v',
    'unstable' : '^',
    'neutral' : 'o',
    'stable' : 'D',
    'strongly stable' : 's',
    'default1' : 'o',
    'default2' : 's'
}

HEIGHTS = [6, 10, 20, 32, 106] # Heights that we are concerned with for plotting, in meters. 80m is left out here.
ZVALS = np.linspace(0.,130.,400) # Linspace for plotting heights

def change_luminosity(color, amount=1):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def bar_stability(df, summary, size, saveto, poster):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size)
    stability_percents = 100 * df['stability'].value_counts(normalize = True)
    if summary['stability_classes'] == 4:
        ax.bar(
            x = ['Unstable\n'+r'$Ri_b<-0.1$','Neutral\n'+r'$-0.1\leq Ri_b<0.1$','Stable\n'+r'$0.1\leq Ri_b<0.25$','Strongly Stable\n'+r'$0.25\leq Ri_b$'],
            height = [stability_percents['unstable'], stability_percents['neutral'], stability_percents['stable'], stability_percents['strongly stable']],
            color = [COLORS['unstable'], COLORS['neutral'], COLORS['stable'], COLORS['strongly stable']]
        )
        for i, sc in enumerate(['unstable', 'neutral', 'stable', 'strongly stable']):
            ax.text(i, stability_percents[sc] - 2, f'{"N=" if i == 0 else ""}{len(df[df["stability"] == sc])}', ha='center', va='center')
    elif summary['stability_classes'] == 3:
        ax.bar(
            x = ['Unstable\n'+r'$Ri_b<-0.1$','Neutral\n'+r'$-0.1\leq Ri_b<0.1$','Stable\n'+r'$0.1\leq Ri_b$'],
            height = [stability_percents['unstable'], stability_percents['neutral'], stability_percents['stable']],
            color = [COLORS['unstable'], COLORS['neutral'], COLORS['stable']]
        )
        for i, sc in enumerate(['unstable', 'neutral', 'stable']):
            ax.text(i, stability_percents[sc] - 3, f'{"N=" if i == 0 else ""}{len(df[df["stability"] == sc])}', ha='center', va='center')
    else:
        print(f"Cannot handle {summary['stability_classes']} stability classes in plot 'bar_stability'")
    if poster:
        ax.set_title('Frequency of Wind Data Sorted by \nBulk Richardson Number Thermal Stability Classification')
    ax.set_ylabel('Proportion of Data (%)')
    ax.grid(axis = 'y', linestyle = '-', alpha = 0.75)
    plt.savefig(saveto, bbox_inches = 'tight')
    return

def annual_profiles(df, summary, size, saveto, poster):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(1, 2, figsize = size, sharey = True)
    if summary['stability_classes'] == 4:
        stabilities = ['unstable', 'neutral', 'stable', 'strongly stable']
    elif summary['stability_classes'] == 3:
        stabilities = ['unstable', 'neutral', 'stable']
    else:
        print(f"Cannot handle {summary['stability_classes']} stability classes in plot 'annual_profiles'")
    for i, tc in enumerate(['open', 'complex']):
        ax = axs[i]
        dft = df[df['terrain'] == tc]
        for sc in stabilities:
            short = ''.join([sub[0] for sub in sc.title().split(' ')])
            dfs = dft[dft['stability'] == sc]
            means = dfs[[f'ws_{h}m' for h in HEIGHTS]].mean(axis = 0).values
            mult, wsc = stats.power_fit(HEIGHTS, means)
            ax.plot(mult * ZVALS**wsc, ZVALS, color = change_luminosity(COLORS[sc], 0.85), zorder = 0)
            ax.scatter(means, HEIGHTS, label = r'{sc}: $u(z)={a:.2f}z^{{{b:.3f}}}$'.format(sc=short,a=mult,b=wsc), color = COLORS[sc], zorder = 5, s = 75*3**poster, marker = MARKERS[sc])
        ax.set_xlabel('Mean Wind Speed (m/s)')
        if i == 0: ax.set_ylabel('Height (m)')
        if poster:
            tc_title = (r'Open Terrain (${openL}-{openR}\degree$ at {h}m)'.format(openL = int(135 - summary['terrain_window_width_degrees']/2), openR = int(135 + summary['terrain_window_width_degrees']/2), h = summary['terrain_wind_height_meters'])
                    if tc == 'open'
                    else r'Complex Terrain (${complexL}-{complexR}\degree$ at {h}m)'.format(complexL = int(315 - summary['terrain_window_width_degrees']/2), complexR = int(315 + summary['terrain_window_width_degrees']/2), h = summary['terrain_wind_height_meters'])
                )
            ax.set_title(tc_title)
        ax.legend(loc = 'upper left', fancybox = False, shadow = False, ncol = 1)
    if poster:
        fig.suptitle('Annual Profiles of Wind Speed, by Terrain and Stability')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight')
    return

def wse_histograms(df, summary, size, saveto, poster):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size)
    plt.savefig(saveto, bbox_inches = 'tight')
    return

def veer_profiles(df, summary, size, saveto, poster):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(1, 2, figsize = size, sharey = True)
    if summary['stability_classes'] == 4:
        stabilities = ['unstable', 'neutral', 'stable', 'strongly stable']
    elif summary['stability_classes'] == 3:
        stabilities = ['unstable', 'neutral', 'stable']
    else:
        print(f"Cannot handle {summary['stability_classes']} stability classes in plot 'annual_profiles'")
    for i, tc in enumerate(['open', 'complex']):
        ax = axs[i]
        dft = df[df['terrain'] == tc]
        for sc in stabilities:
            dfs = dft[dft['stability'] == sc]
            means = [polar.unit_average_direction(dfs[f'wd_{h}m']) for h in HEIGHTS]
            ax.plot(means, HEIGHTS, color = change_luminosity(COLORS[sc], 0.85), zorder = 0)
            ax.scatter(means, HEIGHTS, label = sc.title(), zorder = 5, s = 75*3**poster, marker = MARKERS[sc], facecolors = 'none', edgecolors = COLORS[sc], linewidths = 1.5)
        ax.set_xlabel('Mean Wind Direction (degrees)')
        if i == 0:
            ax.set_ylabel('Height (m)')
            ax.legend(loc = 'lower right', fancybox = False, shadow = False, ncol = 1)
        if poster:
            tc_title = (r'Open Terrain (${openL}-{openR}\degree$ at {h}m)'.format(openL = int(135 - summary['terrain_window_width_degrees']/2), openR = int(135 + summary['terrain_window_width_degrees']/2), h = summary['terrain_wind_height_meters'])
                    if tc == 'open'
                    else r'Complex Terrain (${complexL}-{complexR}\degree$ at {h}m)'.format(complexL = int(315 - summary['terrain_window_width_degrees']/2), complexR = int(315 + summary['terrain_window_width_degrees']/2), h = summary['terrain_wind_height_meters'])
                )
            ax.set_title(tc_title)
    if poster:
        fig.suptitle('Annual Profiles of Wind Direction, by Terrain and Stability')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight')

def tod_wse(df, summary, size, saveto, poster):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    # Two hourly plots, with shared x axis. One open, one complex.
    # In each plot: each stability class plotted, with a sine fit (print results, maybe display in legend)
    #   Scatter plot with error bars. Find a way to make it look good with all 4 stability (e.g. have them slightly offset horizontally so as to be side-by-side at each hour)
    fig, ax = plt.subplots(figsize = size)
    plt.savefig(saveto, bbox_inches = 'tight')
    return

def data_gaps(df, summary, size, saveto, poster):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size)
    plt.savefig(saveto, bbox_inches = 'tight')
    return

def terrain_breakdown(df, summary, size, saveto, poster):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size)
    plt.savefig(saveto, bbox_inches = 'tight')
    return

ALL_PLOTS = {
    'bar_stability': ('Stability Frequency Bar Plot', bar_stability, (8,6)),
    'annual_profiles' : ('Annual Wind Profiles with Fits, by Terrain', annual_profiles, (13,6)),
    'wse_histograms' : ('Histograms of WSE, by Stability, including Terrain', wse_histograms, (12,11)),
    'veer_profiles' : ('Wind direction profiles, by Terrain', veer_profiles, (13,6)),
    'tod_wse' : ('Time of Day Plots of WSE, by Terrain, including Stability & Fits', tod_wse, (13,18)),
    'data_gaps' : ('Data Gap Visualization', data_gaps, (13,8)),
    'terrain_breakdown' : ('Breakdown of Terrain Characterizations, by Month', terrain_breakdown, (7,10))
}

def list_possible_plots():
    print('Possible plots to generate:')
    for tag in ALL_PLOTS.keys():
        print(f'\t{tag}')

def generate_plots(df: pd.DataFrame, savedir: str, summary: dict, which: list = ALL_PLOTS.keys(), poster: bool = False, **kwargs):
    plt.rcParams['font.size'] = 26 if poster else 14
    plt.rcParams['font.family'] = 'sans-serif' if poster else 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans' if poster else 'stix'
    print(f'Generating final plots in {"poster" if poster else "paper"} mode')
    not_generated = list(ALL_PLOTS.keys())
    fig_savedir = f'{savedir}/{"P" if poster else ""}{summary["_rules_chksum"]}'
    os.makedirs(fig_savedir, exist_ok = True)
    for tag in which:
        long, plotter, size = ALL_PLOTS[tag]
        print(f"Generating plot {tag}: '{long}'")
        if poster: size = (2*size[0],2*size[1]) # higher quality for poster scaling
        plotter(df = df, summary = summary, size = size, saveto = f'{fig_savedir}/{tag}.png', poster = poster)
        not_generated.remove(tag)
    if len(not_generated) != 0:
        print(f'Plots not generated: {not_generated}') 
    print(f'Finished generating plots. Final plots saved to:\n\t{fig_savedir}/')
    print('Rules used in analysis to create plot data:')
    for key, val in summary.items():
        if key[0] != '_':
            print(f'\t{key} = {val}')
