import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import numpy as np
import os
import windprofiles.lib.stats as stats
import windprofiles.lib.polar as polar
from windprofiles.analyze import get_monthly_breakdown, get_correlations
from windprofiles.lib.other import time_to_hours
from windprofiles.plotting import change_luminosity
import datetime
from astral import LocationInfo
from astral.sun import sun
import windrose
from kcc_definitions import LATITUDE, LONGITUDE

FIGTAB = "C:/Users/22wal/Documents/GLWind/results/fig_tabular"

ROSE_BOOM = 2
DRMSBOOM = 6 # Boom number for directional RMS comparison

neatline = False

COLORS_POSTER = {
    'open' : '#1f77b4',
    'complex' : '#ff7f0e',
    'other' : 'tab:green',
    'unstable' : '#ed2857',
    'neutral' : '#fcc749',
    'stable' : '#0aa179',
    'strongly stable' : '#119dab',
    'default1' : '#7f7f7f',
    'default2' : '#2ca02c',
    'available' : 'b',
    'unavailable' : 'r'
}

COLORS_FORMAL = {
    'open' : 'tab:blue',
    'complex' : 'tab:orange',
    'other' : 'tab:green',
    'unstable' : 'tab:red',
    'neutral' : '#3b50d6',
    'stable' : '#9b5445',
    'strongly stable' : 'tab:green',
    'default1' : 'tab:blue',
    'default2' : 'tab:orange',
    'available' : 'b',
    'unavailable' : 'r'
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

BOOMS = [1, 2, 3, 4, 6]
HEIGHTS = [6., 10., 20., 32., 106.] # Heights that we are concerned with for plotting, in meters. 80m is left out here.

BOOMS_GAPS = [1, 2, 3, 4, 5, 6]
HEIGHTS_GAPS = [6., 10., 20., 32., 80., 106.] # Every height (HEIGHTS but with 80 as well), for the data gaps visualization

ZVALS = np.linspace(0.,157.5,400) # Linspace for plotting heights
UVALS = np.linspace(0, 17, 360) # Linspace for plotting wind speed distributions
TERRAINS = ['open', 'complex']
LOCATION = LocationInfo(name = 'KCC tower', region = 'IA, USA', timezone = 'US/Central', latitude = LATITUDE, longitude = LONGITUDE)

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SEASONS = {
    'fall' : ['Sep', 'Oct', 'Nov'],
    'winter' : ['Dec', 'Jan', 'Feb'],
    'spring' : ['Mar', 'Apr', 'May'],
    'summer' : ['Jun', 'Jul', 'Aug']
}
SEASON_STRINGS = {
    'fall' : 'Fall (Sep-Nov)',
    'winter' : 'Winter (Dec-Feb)',
    'spring' : 'Spring (Mar-May)',
    'summer' : 'Summer (Jun-Aug)'
}
CENTERDATES = { # Solstices/equinoxes in 2018
    'fall' : datetime.date(2018, 9, 22),
    'winter' : datetime.date(2018, 12, 21),
    'spring' : datetime.date(2018, 3, 20),
    'summer' : datetime.date(2018, 6, 21)
}

GAPS_LINEAR = False # Show data_gaps plot with a linear y scale?
GAPS_LOGARITHMIC = True # Show data_gaps plot with a logarithmic y scale?
if GAPS_LINEAR and GAPS_LOGARITHMIC:
    raise('Cannot have both log and linear scale')
DISTS_BY_TERRAIN = True # Show separate terrain classes in speed_distributions and pti_distributions?

CORRELATION_VARIABLES = [f'ws_{b}' for b in BOOMS_GAPS] + [f't_{b}' for b in BOOMS_GAPS] + [f'p_{b}' for b in BOOMS_GAPS] + [f'rh_{b}' for b in BOOMS_GAPS] + [f'pti_{b}' for b in BOOMS_GAPS] + [f'drms_{b}' for b in BOOMS_GAPS] + ['vpt_2', 'vpt_6', 'vpt_lapse', 'alpha', 'Ri_bulk']

def bar_stability(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')
    stability_percents = 100 * df['stability'].value_counts(normalize = True)
    if summary['stability_classes'] == 4:
        ax.bar(
            x = ['Unstable\n'+r'$Ri_b<-0.1$','Neutral\n'+r'$-0.1\leq Ri_b<0.1$','Stable\n'+r'$0.1\leq Ri_b<0.25$','Strongly Stable\n'+r'$0.25\leq Ri_b$'],
            height = [stability_percents['unstable'], stability_percents['neutral'], stability_percents['stable'], stability_percents['strongly stable']],
            color = [COLORS['unstable'], COLORS['neutral'], COLORS['stable'], COLORS['strongly stable']]
        )
        for i, sc in enumerate(['unstable', 'neutral', 'stable', 'strongly stable']):
            ax.text(i, stability_percents[sc] + 1.25, f'{"N=" if i == 0 else ""}{len(df[df["stability"] == sc])}', ha='center', va='center')
    elif summary['stability_classes'] == 3:
        ax.bar(
            x = ['Unstable\n'+r'$Ri_b<-0.1$','Neutral\n'+r'$-0.1\leq Ri_b<0.1$','Stable\n'+r'$0.1\leq Ri_b$'],
            height = [stability_percents['unstable'], stability_percents['neutral'], stability_percents['stable']],
            color = [COLORS['unstable'], COLORS['neutral'], COLORS['stable']]
        )
        for i, sc in enumerate(['unstable', 'neutral', 'stable']):
            ax.text(i, stability_percents[sc] + 1.25, f'{"N=" if i == 0 else ""}{len(df[df["stability"] == sc])}', ha='center', va='center')
    else:
        raise Exception(f"Cannot handle {summary['stability_classes']} stability classes in plot 'bar_stability'")
    if poster:
        ax.set_title('Frequency of Wind Data Sorted by \nBulk Richardson Number Thermal Stability Classification')
    ax.set_ylim(0, max(stability_percents) + 3)
    ax.set_ylabel('Proportion of Data (%)')
    ax.grid(axis = 'y', linestyle = '-', alpha = 0.75)
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def annual_profiles(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(1, 2, figsize = size, sharey = True, linewidth = 5*poster*neatline, edgecolor = 'k')
    if summary['stability_classes'] == 4:
        stabilities = ['unstable', 'neutral', 'stable', 'strongly stable']
    elif summary['stability_classes'] == 3:
        stabilities = ['unstable', 'neutral', 'stable']
    else:
        raise Exception(f"Cannot handle {summary['stability_classes']} stability classes in plot 'annual_profiles'")
    
    out_table = pd.DataFrame(index = HEIGHTS)

    for i, tc in enumerate(TERRAINS):
        ax = axs[i]
        dft = df[df['terrain'] == tc]

        for sc in stabilities:
            short = sc.title()
            ax.set_ylim((-7.5,167.5))
            dfs = dft[dft['stability'] == sc]

            zorder = 0
            def plot_fit(booms, heights, linestyle, scatter = False): # linestyle can be 'solid', 'dashed', 'dotted', 'dashdot', or a parameterization tuple
                nonlocal zorder
                means = dfs[[f'ws_{b}' for b in booms]].mean(axis = 0).values
                mult, wsc = stats.power_fit(heights, means)
                ax.plot(mult * ZVALS**wsc, ZVALS, color = change_luminosity(COLORS[sc], 0.85), zorder = zorder, linestyle = linestyle)
                zorder += 2
                if scatter: # enable this for the full line
                    ax.scatter(means, heights, label = r'{sc}: $u(z)={a:.2f}z^{{{b:.3f}}}$'.format(sc=short,a=mult,b=wsc), color = COLORS[sc], zorder = 6, s = 75, marker = MARKERS[sc])
                    out_table[f"u_{tc}_{sc.replace(" ","-")}"] = means

            plot_fit(booms = BOOMS, heights = HEIGHTS, linestyle = 'solid', scatter = True) # full (5-height) fit line

            #plot_fit(heights = [6, 10], linestyle = 'dashed') # 2 lowest heights only "Ex 1"
            #plot_fit(heights = [10, 106], linestyle = 'dashed') # 2 key heights (10 & 106 meters) only "Ex 2"
            #plot_fit(heights = [6, 10, 20, 32], linestyle = 'dashed') # all heights except for 106 meters (6 - 32) "Ex 3"
            
        out_table.to_csv(os.path.join(FIGTAB, "annual_profiles_fig8.csv"), index_label="height", float_format = "%.6f")

        ax.set_xlabel('Mean Wind Speed (m/s)')
        ax.set_title(tc.title(), loc = 'right', x = 0.98, y = 0.02)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        if i == 0: ax.set_ylabel('Height (m)')
        if poster:
            tc_title = (r'Open Terrain (${openL}-{openR}\degree$ at {b})'.format(openL = int(135 - summary['terrain_window_width_degrees']/2), openR = int(135 + summary['terrain_window_width_degrees']/2), b = summary['terrain_wind_boom'])
                    if tc == 'open'
                    else r'Complex Terrain (${complexL}-{complexR}\degree$ at {b})'.format(complexL = int(315 - summary['terrain_window_width_degrees']/2), complexR = int(315 + summary['terrain_window_width_degrees']/2), b = summary['terrain_wind_boom'])
                )
            ax.set_title(tc_title)
        ax.legend(loc = 'upper left')
    if poster:
        fig.suptitle('Annual Profiles of Wind Speed, by Terrain and Stability')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def wse_histograms(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    if summary['stability_classes'] == 4:
        stabilities = ['unstable', 'neutral', 'stable', 'strongly stable']
        nrows = 2
        ncols = 2
    elif summary['stability_classes'] == 3:
        stabilities = ['unstable', 'neutral', 'stable']
        nrows = 1
        ncols = 3
        size = (size[0] * 1.5, size[1] * 0.7)
    else:
        raise Exception(f"Cannot handle {summary['stability_classes']} stability classes in plot 'wse_histograms'")
    fig, axs = plt.subplots(nrows, ncols, figsize = size, sharex = (nrows > 1), linewidth = 5*poster*neatline, edgecolor = 'k')
    for i, ax in enumerate(fig.axes):
        sc = stabilities[i]
        dfs = df[df['stability'] == sc]
        for j, tc in enumerate(TERRAINS):
            dft_alpha = dfs.loc[dfs['terrain'] == tc, 'alpha']
            # density = True makes it such that the area under the histogram integrates to 1
            ax.hist(dft_alpha, bins = 35, density = True, alpha = 0.5, color = COLORS[tc], edgecolor = 'k', range = (-0.3, 1.2), label = f'{tc.title()} terrain')
            mean = dft_alpha.mean()
            median = dft_alpha.median()
            std = dft_alpha.std()
            skew = dft_alpha.skew()
            textstr = '\n'.join((
                tc.title(),
                f'mean: {mean:.2f}',
                f'median: {median:.2f}',
                f'stdev: {std:.2f}',
                f'skew: {skew:.2f}'
            ))
            ax.text(0.75, 0.77-0.35*j, textstr, transform = ax.transAxes, verticalalignment = 'top', bbox = dict(boxstyle = 'round', facecolor = 'none', edgecolor = COLORS[tc], linewidth = 2))
            if details:
                print(f'  {tc} {sc}:')
                print(f'\tMean: {mean:.2f}')
                print(f'\tMedian: {median:.2f}')
                print(f'\tStandard deviation: {std:.2f}')
                print(f'\tMed Ri_b: {dfs.loc[dfs["terrain"] == tc, "Ri_bulk"].median():.2f}')
        if i == ncols - 1:
            ax.legend(loc = 'upper right')
        if (nrows == 2 and i >= ncols):
            ax.set_xlabel(r'$\alpha$')
        if (i+1) % ncols == 1:
            ax.set_ylabel('Probability Density')
        ax.set_title(sc.title(), loc = 'left', x = 0.025, y = 0.9125)
        ax.vlines(x = [dfs.loc[dfs['terrain'] == tc, 'alpha'].median() for tc in TERRAINS], ymin = 0, ymax = ax.get_ylim()[1], colors = [change_luminosity(COLORS[tc], 1.5) for tc in TERRAINS], alpha = 0.75, linestyle = 'dashed', linewidth = 4)
    if poster:
        fig.suptitle(r'Wind Shear Exponent Distributions, by Terrain and Stability')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def veer_profiles(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(1, 2, figsize = size, sharey = True, linewidth = 5*poster*neatline, edgecolor = 'k')
    if summary['stability_classes'] == 4:
        stabilities = ['unstable', 'neutral', 'stable', 'strongly stable']
    elif summary['stability_classes'] == 3:
        stabilities = ['unstable', 'neutral', 'stable']
    else:
        raise Exception(f"Cannot handle {summary['stability_classes']} stability classes in plot 'annual_profiles'")
    
    out_table = pd.DataFrame(index = HEIGHTS)
    
    for i, tc in enumerate(TERRAINS):
        ax = axs[i]
        dft = df[df['terrain'] == tc]
        for sc in stabilities:
            dfs = dft[dft['stability'] == sc]
            means = [polar.unit_average_direction(dfs[f'wd_{b}']) for b in BOOMS]
            ax.plot(means, HEIGHTS, color = change_luminosity(COLORS[sc], 0.85), zorder = 0)
            ax.scatter(means, HEIGHTS, label = sc.title(), zorder = 5, s = 75, marker = MARKERS[sc], facecolors = 'none', edgecolors = COLORS[sc], linewidths = 1.5)
            out_table[f"wd_{tc}_{sc.replace(" ","-")}"] = means

        out_table.to_csv(os.path.join(FIGTAB, "veer_profiles_fig11.csv"), index_label="height", float_format = "%.6f")

        ax.set_xlabel('Mean Wind Direction (degrees)')
        ax.set_title(tc.title(), loc = 'right', x = 0.98, y = 0.02)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)

        if i == 0:
            ax.set_xlim(125, 175)
            ax.set_ylabel('Height (m)')
        elif i == 1:
            ax.set_xlim(295, 345)
            ax.legend(loc = 'upper left')
        if poster:
            tc_title = (r'Open Terrain (${openL}-{openR}\degree$ at {b})'.format(openL = int(135 - summary['terrain_window_width_degrees']/2), openR = int(135 + summary['terrain_window_width_degrees']/2), b = summary['terrain_wind_boom'])
                    if tc == 'open'
                    else r'Complex Terrain (${complexL}-{complexR}\degree$ at {b})'.format(complexL = int(315 - summary['terrain_window_width_degrees']/2), complexR = int(315 + summary['terrain_window_width_degrees']/2), b = summary['terrain_wind_boom'])
                )
            ax.set_title(tc_title)
    if poster:
        fig.suptitle('Annual Profiles of Wind Direction, by Terrain and Stability')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)

def tod_wse(df, cid, summary, size, saveto, poster, details):
    OFFSET = 0.15
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(nrows = summary['stability_classes'], ncols = 1, figsize = size, sharex = True, linewidth = 5*poster*neatline, edgecolor = 'k')
    if summary['stability_classes'] == 4:
        stabilities = ['unstable', 'neutral', 'stable', 'strongly stable']
    elif summary['stability_classes'] == 3:
        stabilities = ['unstable', 'neutral', 'stable']
    else:
        raise Exception(f"Cannot handle {summary['stability_classes']} stability classes in plot 'tod_wse'")
    
    out_table = pd.DataFrame(index = range(24))

    for i, ssn in enumerate(['fall','winter','spring','summer']):
        ax = axs[i]
        ax.set_ylim(0,0.725)
        mons = SEASONS[ssn]
        monnums = [MONTHS.index(m)+1 for m in mons]
        dfs = df[df['time'].dt.month.isin(monnums)]
        for j, tc in enumerate(TERRAINS):
            dft = dfs[dfs['terrain'] == tc]
            hourly_wse = [dft[dft['time'].dt.hour == hr]['alpha'].reset_index(drop=True) for hr in range(24)]            
            hourly_wse.append(dft[dft['time'].dt.hour == 0]['alpha'].reset_index(drop=True)) # have 0 at both the start and end
            med_wse = [wse.median() for wse in hourly_wse]
            std_wse = [wse.std() for wse in hourly_wse]

            out_table[f"alpha_{tc}_{ssn}"] = med_wse[:-1]
            out_table[f"err_{tc}_{ssn}"] = std_wse[:-1]

            ax.errorbar(x = np.array(range(25))+OFFSET*j, y = med_wse, yerr = std_wse, color = COLORS[tc], fmt = MARKERS[tc], markersize = 12, label = r'$\alpha$ ({terr})'.format(terr = tc.title()))
            fitsine, params = stats.fit_sine(range(24), med_wse[:24], std_wse[:24], fix_period=True)
            if details:
                print(f'\t{ssn} alpha = {params[0]:.4f} * sin({params[1]:.4f} * t + {params[2]:.4f}) + {params[3]:.4f}')
            xplot = np.linspace(0, 24, 120)
            ax.plot(xplot+OFFSET*j, fitsine(xplot), color = COLORS[tc], linestyle = 'dashed', alpha = 0.75)
            fit_label = r'$\alpha = {A:.3f}\cdot\sin ({f:.3f}t+{b:.3f})+{h:.3f}$'.format(A=params[0], f=params[1], b = params[2], h = params[3])
            ax.text(0.49, 0.84-0.11*j, s = fit_label, size = 'large', horizontalalignment = 'center', verticalalignment = 'top', color = change_luminosity(COLORS[tc], 1.15), transform = ax.transAxes)
        
        out_table.to_csv(os.path.join(FIGTAB, "tod_wse_fig10.csv"), index_label="hour", float_format="%.6f")

        ax.set_ylabel(r'$\alpha$')
        ax.set_title(SEASON_STRINGS[ssn], loc = 'center', y = 0.85)
        s = sun(LOCATION.observer, date = CENTERDATES[ssn], tzinfo = LOCATION.timezone)
        ax.vlines([time_to_hours(s['sunrise']), time_to_hours(s['sunset'])], linestyle = 'dashed', ymin = ax.get_ylim()[0], ymax = ax.get_ylim()[1], color = 'green')
        if i == 0:
            ax.legend(loc = 'upper left')
        elif i == 3:
            major_tick_locations = np.array(range(0,25,6)) + OFFSET*j/2
            major_tick_labels = [6*i for i in range(4)] + [0]
            ax.set_xticks(ticks = major_tick_locations, labels = major_tick_labels, minor = False)
            ax.set_xticks(np.array(range(24)) + OFFSET*j/2, range(24), minor = True, size = 10)
            ax.set_xlabel('Local time (hours)')
    if poster: fig.suptitle('Wind Shear Exponent Medians by Time of Day', y = 1)
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def data_gaps(df, cid, summary, size, saveto, poster, details):
    # Full range of data after quality control, to show what specific times are covered by the analysis
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize=size, linewidth=5*poster, edgecolor='k')

    GAP_HEIGHT = 4 if GAPS_LOGARITHMIC else 0

    period = summary['resampling_window_minutes']
    start = df['time'].min()
    end = df['time'].max()
    all_times = pd.date_range(start, end, freq=f'{period}min')

    # Identify missing timestamps
    gaps = all_times[~all_times.isin(df['time'])]
    ax.scatter(gaps, [GAP_HEIGHT for _ in gaps], s=2, c=COLORS['unavailable'])

    # Plot available data at each height
    for i, (b, h) in enumerate(zip(BOOMS_GAPS, HEIGHTS_GAPS), 1):
        available = df[~pd.isna(df[f'ws_{b}'])]['time']
        ax.scatter(available, [(h if GAPS_LINEAR or GAPS_LOGARITHMIC else i) for _ in available], s=2, c=COLORS['available'])

    # Set y-axis labels and grid
    if GAPS_LOGARITHMIC:
        ax.set_yscale('log')
        ax.set_ylim(3, 140)
    if GAPS_LINEAR or GAPS_LOGARITHMIC:
        ax.set_yticks([GAP_HEIGHT] + HEIGHTS_GAPS)
    else:
        ax.set_yticks(range(len(HEIGHTS_GAPS) + 1))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_yticklabels(['No Data'] + [str(int(h)) for h in HEIGHTS_GAPS])
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylabel('Boom Height (m)')

    # Set x-axis grid at the start of each month
    month_starts = pd.date_range(start=start, end=end, freq='MS')  # 'MS' = Month Start
    ax.set_xticks(month_starts)
    ax.set_xticklabels([f"{MONTHS[d.month - 1]}{(' ' + str(d.year)) if (i==0 or month_starts[i].year != month_starts[i-1].year) else ''}" for i, d in enumerate(month_starts)])
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('Time')

    fig.tight_layout()

    #plt.show()

    plt.savefig(saveto, bbox_inches='tight', edgecolor=fig.get_edgecolor())

    return

def terrain_breakdown(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')
    breakdown, proportions = get_monthly_breakdown(df, 'terrain')
    last = pd.Series(np.zeros(12, dtype = int))
    for tc in ['open', 'complex', 'other']:
        num = breakdown[tc].reset_index(drop = True)
        perc = proportions[tc].reset_index(drop = True)
        ax.bar(MONTHS, num, bottom = last, label = tc.title(), color = COLORS[tc])
        for i in range(len(MONTHS)):
            plt.text(i, last.iloc[i] + num.iloc[i]/2, f'{100*perc.iloc[i]:.1f}%', ha = 'center', va = 'center')
        last += num
    top_offset = np.max(last) / 150
    for i in range(12):
        plt.text(i, last.iloc[i] + top_offset, f"{'N=' if i == 0 else ''}{last.iloc[i]}", ha = 'center')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Data Points')
    ax.legend(bbox_to_anchor = (0.575,0.975))
    if poster: fig.suptitle(f'Terrain Breakdown Based on Boom {summary["terrain_wind_boom"]} Wind Directions')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def windrose_comparison(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(ncols=2, figsize=size, linewidth=5*poster, edgecolor='k', subplot_kw={'projection': 'windrose'})    
    
    # Custom legend labels
    speed_bins = [0.,1.5,3.,4.5,6.,7.5,9.,1000] # 1000 in place of np.inf (unrealistically high speed value; np.inf gives a RuntimeWarning even though it works)
    cmap = cm.rainbow  # Color map used for bins (note: append suffix _r to use reversed version)
    # Good cmaps: jet, jet_r, gist_rainbow, coolwarm, coolwarm_r, plasma_r, viridis_r, rainbow_r, rainbow
    colors = [cmap(i / len(speed_bins)) for i in range(len(speed_bins))]

    # First, KCC
    speeds_kcc = df[f'ws_{ROSE_BOOM}']
    directions_kcc = df[f'wd_{ROSE_BOOM}']
    axs[0].bar(directions_kcc, speeds_kcc, normed=True, opening=1.0,
               bins=speed_bins, edgecolor='white', linewidth = 0.1, colors=colors, nsector=180) # TMP 36
    axs[0].set_title('KCC', y=1.075)

    # Now, CID
    speeds_cid = cid['ws_0']
    directions_cid = cid['wd_0']
    axs[1].bar(directions_cid, speeds_cid, normed=True, opening=1.0,
               bins=speed_bins, edgecolor='white', linewidth = 0.1, colors=colors, nsector=36)
    axs[1].set_title('CID', y=1.075)

    # Ensure both plots have the same radial ticks by getting ticks from one axis
    tick_values = np.linspace(0, axs[0].get_ylim()[1], num=5)
    for ax in axs:
        ax.set_yticks(tick_values, ['' if value == 0 else f'{value:.1f}%' for value in tick_values])
    
    # Decrease the gap between the subplots
    fig.subplots_adjust(wspace=-0.55)  # Adjust the value as needed
    
    # Custom legend - moved to the left side
    legend_labels = [f"{speed_bins[i]:.1f}-{speed_bins[i+1]:.1f} m/s" for i in range(len(speed_bins)-2)] + [f">{speed_bins[-2]:.1f} m/s"]
    patches = [plt.Line2D([0], [0], color=colors[i], lw=5) for i in range(len(legend_labels))]
    fig.legend(patches, legend_labels, title="Wind Speed (m/s)", loc='center left', bbox_to_anchor=(-0.1, 0.5), ncol=1)
    
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches='tight', edgecolor=fig.get_edgecolor())
    return

def pti_profiles(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(1, 2, figsize = size, sharey = True, linewidth = 5*poster*neatline, edgecolor = 'k')
    if summary['stability_classes'] == 4:
        stabilities = ['unstable', 'neutral', 'stable', 'strongly stable']
    elif summary['stability_classes'] == 3:
        stabilities = ['unstable', 'neutral', 'stable']
    else:
        raise Exception(f"Cannot handle {summary['stability_classes']} stability classes in plot 'annual_profiles'")
    
    out_table = pd.DataFrame(index = HEIGHTS)
    
    for i, tc in enumerate(TERRAINS):
        ax = axs[i]
        dft = df[df['terrain'] == tc]
        for sc in stabilities:
            dfs = dft[dft['stability'] == sc]
            means = dfs[[f'pti_{b}' for b in BOOMS]].mean(axis = 0).values
            ax.plot(means, HEIGHTS, color = change_luminosity(COLORS[sc], 0.85), zorder = 0)
            ax.scatter(means, HEIGHTS, label = sc.title(), zorder = 5, s = 75, marker = MARKERS[sc], facecolors = 'none', edgecolors = COLORS[sc], linewidths = 1.5)
            out_table[f"ti_{tc}_{sc.replace(" ","-")}"] = means

        out_table.to_csv(os.path.join(FIGTAB, "ti_profiles_fig12.csv"), index_label="height", float_format = "%.6f")

        ax.set_xlabel(r'$TI$')
        if summary['turbulence_method_local']:
            ax.set_xlim(0, 0.25)
        else:
            ax.set_xlim(0, 0.2)
        ax.set_title(tc.title(), loc = 'right', x = 0.98, y = 0.02)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.set_ylabel('Height (m)')
            ax.legend(loc = 'upper right')
        if poster:
            tc_title = (r'Open Terrain (${openL}-{openR}\degree$ at {b})'.format(openL = int(135 - summary['terrain_window_width_degrees']/2), openR = int(135 + summary['terrain_window_width_degrees']/2), b = summary['terrain_wind_boom'])
                    if tc == 'open'
                    else r'Complex Terrain (${complexL}-{complexR}\degree$ at {b})'.format(complexL = int(315 - summary['terrain_window_width_degrees']/2), complexR = int(315 + summary['terrain_window_width_degrees']/2), b = summary['terrain_wind_boom'])
                )
            ax.set_title(tc_title) 
    if poster:
        fig.suptitle(r'Annual Profiles of Turbulence Intensity, by Terrain and Stability')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)

def speed_distributions(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(nrows = len(HEIGHTS), ncols = 1, sharex = True, figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')

    N = len(HEIGHTS)
    for i, (b, h) in enumerate(zip(BOOMS, HEIGHTS),1):
        ax = axs[N-i]
        if details:
            print(f'At {h} meters')
        if DISTS_BY_TERRAIN:
            for j, tc in enumerate(TERRAINS):
                ws_hm = df[df['terrain'] == tc][f'ws_{b}']
                ws_noinf = ws_hm.replace([np.inf,-np.inf], np.nan).dropna()
                weib, [shape, scale] = stats.fit_wind_weibull(ws_noinf)
                if details:
                    print('\t'+tc.title())
                    print(f'\t\tShape parameter k = {shape:.4f}')
                    print(f'\t\tScale parameter lambda = {scale:.4f}')
                ax.hist(x = ws_noinf, bins = 60, density = True, alpha = 0.4, color = COLORS[tc], edgecolor = 'k', range = (0, 20), label = tc.title())
                ax.plot(UVALS, weib(UVALS), color = change_luminosity(COLORS[tc], 0.85))
                fit_label = r'$k={k:.2f},~\lambda={lamb:.2f}$'.format(k = shape, lamb = scale)
                ax.text(0.99, 0.815-0.135*j, s = fit_label, size = 'large', horizontalalignment = 'right', verticalalignment = 'top', color = change_luminosity(COLORS[tc], 1.15), transform = ax.transAxes)
                if i == 1:
                    ax.legend(loc = 'lower right')
        else:
            ws_hm = df[f'ws_{b}']
            ws_noinf = ws_hm.replace([np.inf,-np.inf], np.nan).dropna()
            weib, [shape, scale] = stats.fit_wind_weibull(ws_noinf)
            if details:
                print(f'Shape parameter k = {shape:.4f}')
                print(f'Scale parameter lambda = {scale:.4f}')
            ax.hist(x = ws_noinf, bins = 60, density = True, alpha = 0.4, color = COLORS['default1'], edgecolor = 'k', range = (0, 20))
            ax.plot(UVALS, weib(UVALS), color = change_luminosity(COLORS['default1'], 0.85))
            fit_label = r'$k={k:.2f},~\lambda={lamb:.2f}$'.format(k = shape, lamb = scale)
            ax.text(0.99, 0.815, s = fit_label, size = 'large', horizontalalignment = 'right', verticalalignment = 'top', transform = ax.transAxes)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f'{h} meters', loc = 'right', x = 0.99, y = 0.825)
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 0.265)
        if i == 1:
            ax.set_xlabel('Wind Speed (m/s)')
            ax.set_ylabel('Probability Density')
    if poster:
        fig.suptitle('Wind Speed Distributions with Best-Fit Weibull Curves Overlaid')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def pti_distributions(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, axs = plt.subplots(nrows = len(HEIGHTS), ncols = 1, sharex = True, figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')

    N = len(HEIGHTS)
    for i, (b, h) in enumerate(zip(BOOMS, HEIGHTS),1):
        ax = axs[N-i]

        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 13)

        if DISTS_BY_TERRAIN:
            for j, tc in enumerate(TERRAINS):
                pti_hm = df[df['terrain'] == tc][f'pti_{b}']
                ax.hist(x = pti_hm, bins = 60, density = True, alpha = 0.4, color = COLORS[tc], edgecolor = 'k', range = (0, 0.4), label = tc.title())
                if i == 1:
                    ax.legend(loc = 'lower right')
            ax.vlines(x = [df.loc[df['terrain'] == tc, f'pti_{b}'].mean() for tc in TERRAINS], ymin = 0, ymax = ax.get_ylim()[1], colors = [change_luminosity(COLORS[tc], 1.5) for tc in TERRAINS], alpha = 0.7, linestyle = 'solid', linewidth = 4)
            ax.vlines(x = [df.loc[df['terrain'] == tc, f'pti_{b}'].median() for tc in TERRAINS], ymin = 0, ymax = ax.get_ylim()[1], colors = [change_luminosity(COLORS[tc], 1.5) for tc in TERRAINS], alpha = 0.8, linestyle = 'dashed', linewidth = 4)
        else:
            pti_hm = df[f'pti_{b}']
            ax.hist(x = pti_hm, bins = 60, density = True, alpha = 0.4, color = COLORS['default1'], edgecolor = 'k', range = (0, 0.4))
            ax.vlines(x = [df[f'pti_{b}'].mean()], ymin = 0, ymax = ax.get_ylim()[1], colors = [change_luminosity(COLORS['default1'], 1.5)], alpha = 0.7, linestyle = 'solid', linewidth = 4)
            ax.vlines(x = [df[f'pti_{b}'].median()], ymin = 0, ymax = ax.get_ylim()[1], colors = [change_luminosity(COLORS['default1'], 1.5)], alpha = 0.8, linestyle = 'dashed', linewidth = 4)

        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f'{h} meters', loc = 'right', x = 0.99, y = 0.825)

        if i == 1:
            ax.set_xlabel(r'$TI$')
            ax.set_ylabel('Probability Density')
    if poster:
        fig.suptitle('Pseudo Turbulence Intensity Distributions by Height')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def pti_vs_wse(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')

    o_n_mask = (df['stability'] == 'neutral') & (df['terrain'] == 'open')
    o_n_df = df[o_n_mask]
    etc_df = df[~o_n_mask]

    ax.scatter(etc_df['alpha'], etc_df['pti_6'], s = 0.5, c = COLORS['default1'], label = 'All Other Data')
    ax.scatter(o_n_df['alpha'], o_n_df['pti_6'], s = 2, c = COLORS['default2'], label = 'Open Terrain + Neutral Stability')
    ax.legend()

    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.025, 0.475)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$TI$ (106 meters)')

    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def pti_vs_rib(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')

    ax.scatter(df['Ri_bulk'], df['pti_6'], s = 1, c = COLORS['default1'])

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 0.3)

    ax.set_xlabel(r'$Ri_b$')
    ax.set_ylabel(r'$\sigma_M/\overline{M}$ (106 meters)')

    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def pti_vs_drms(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')

    ax.scatter(df[f'pti_{DRMSBOOM}'], df[f'drms_{DRMSBOOM}'], s = 0.2, c = COLORS['default1'])

    ax.set_xlim(0, 0.175)
    ax.set_ylim(0, 5)

    ax.set_xlabel(r'$\sigma_M/\overline{{M}}$ ({H} meters)'.format(H = int(HEIGHTS_GAPS[DRMSBOOM-1])))
    ax.set_ylabel(r'Directional RMS (degrees)')

    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def correlations(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')

    WHICH = [col for col in CORRELATION_VARIABLES+[f'veer{summary["veer_boom_top"]}-{summary["veer_boom_bottom"]}'] if col in df.columns]
    corrs = get_correlations(df, which = WHICH)

    im = ax.imshow(corrs, cmap = 'seismic', vmin = -1., vmax = 1.)
    ax.set_xticks(ticks = range(len(WHICH)), labels = WHICH, rotation = 90)
    ax.set_yticks(ticks = range(len(WHICH)), labels = WHICH)

    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel("correlation (Pearson r)", rotation = -90, va = "bottom")

    if details:
        print('\nCorrelations\n')
        pd.set_option('display.precision',3)
        print(corrs)
        pd.set_option('display.precision',6)

    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def determinations(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    fig, ax = plt.subplots(figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')

    WHICH = [col for col in CORRELATION_VARIABLES+[f'veer{summary["veer_boom_top"]}-{summary["veer_boom_bottom"]}'] if col in df.columns]
    corrs = get_correlations(df, which = WHICH)**2
    
    im = ax.imshow(corrs, cmap = 'plasma', vmin = 0., vmax = 1.)
    ax.set_xticks(ticks = range(len(WHICH)), labels = WHICH, rotation = 90)
    ax.set_yticks(ticks = range(len(WHICH)), labels = WHICH)

    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel("determination (r^2)", rotation = -90, va = "bottom")

    if details:
        print('\nCoefficients of Determination (r^2)\n')
        pd.set_option('display.precision',3)
        print(corrs)
        pd.set_option('display.precision',6)

    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def alpha_corrs(df):

    def get_and_scatter(col, transy, transx = 'linear', alpha_bounds = (0,1)):
        dfx = df[(df['alpha'] < alpha_bounds[1]) & (df['alpha'] > alpha_bounds[0])]
        if col == 'Ri_bulk':
            dfx = dfx[np.abs(dfx['Ri_bulk']) < 2]
        value = stats.rcorrelation(dfx, col, 'alpha', ('linear', transy))
        print(f'alpha ({transy}) vs {col} ({transx}): {value:.3f} (r^2 = {(value**2):.3f})')
        try:
            plt.scatter(stats.TRANSFORMS[transx](dfx[col]), dfx['alpha'], s = 0.1, alpha = 0.5)
            plt.yscale(transy)
            plt.ylabel(f'alpha ({transy})')
            plt.xlabel(f'{col} ({transx})')
            plt.show()
        except:
            plt.close()
            plt.scatter(stats.TRANSFORMS[transx](dfx[col]), stats.TRANSFORMS[transy](dfx['alpha']), s = 0.1, alpha = 0.5)
            plt.ylabel(f'alpha ({transy})')
            plt.xlabel(f'{col} ({transx})')
            plt.show()
        return

    print('\n\nCORRELATION COEFFICIENTS (PEARSON R) OF ALPHA WITH OTHERS:')
    COLS = ['vpt_lapse', 'Ri_bulk'] #+ [f'pti_{b}' for b in BOOMS_GAPS]
    METHS = ['linear', 'log', 'exp', 'inv']
    for col in COLS:
        for meth in METHS:
            get_and_scatter(col, meth)
        #get_and_scatter(col, 'log', 'log')

    print('\n\n')

    return

def veer_histograms(df, cid, summary, size, saveto, poster, details):
    COLORS = COLORS_POSTER if poster else COLORS_FORMAL
    if summary['stability_classes'] == 4:
        stabilities = ['unstable', 'neutral', 'stable', 'strongly stable']
        nrows = 2
        ncols = 2
    elif summary['stability_classes'] == 3:
        stabilities = ['unstable', 'neutral', 'stable']
        nrows = 1
        ncols = 3
        size = (size[0] * 1.5, size[1] * 0.7)
    else:
        raise Exception(f"Cannot handle {summary['stability_classes']} stability classes in plot 'wse_histograms'")
    fig, axs = plt.subplots(nrows, ncols, figsize = size, sharex = (nrows > 1), linewidth = 5*poster*neatline, edgecolor = 'k')
    for i, ax in enumerate(fig.axes):
        sc = stabilities[i]
        dfs_veer = df.loc[df['stability'] == sc, f'veer{summary["veer_boom_top"]}-{summary["veer_boom_bottom"]}']
        ax.hist(dfs_veer, bins = 35, density = True, alpha = 0.9, color = COLORS[sc], edgecolor = 'k', range = (-90, 90))
        mean = dfs_veer.mean()
        median = dfs_veer.median()
        std = dfs_veer.std()
        skew = dfs_veer.skew()
        textstr = '\n'.join((
            f'mean: {mean:.2f}',
            f'median: {median:.2f}',
            f'stdev: {std:.2f}',
            f'skew: {skew:.2f}'
        ))
        ax.text(0.05, 0.8, textstr, transform = ax.transAxes, verticalalignment = 'top', bbox = dict(boxstyle = 'round', facecolor = 'none', edgecolor = COLORS[sc], linewidth = 2))
        if details:
            print(f' {sc}:')
            print(f'\tMean: {mean:.2f}')
            print(f'\tMedian: {median:.2f}')
            print(f'\tStandard deviation: {std:.2f}')
        if (nrows == 2 and i >= ncols):
            ax.set_xlabel(r'$\alpha$')
        if (i+1) % ncols == 1:
            ax.set_ylabel('Probability Density')
        ax.set_title(sc.title(), loc = 'left', x = 0.025, y = 0.9125)
    if poster:
        fig.suptitle(r'Wind Veer Distributions (\+ for veer, - for backing), by Stability')
    fig.tight_layout()
    plt.savefig(saveto, bbox_inches = 'tight', edgecolor = fig.get_edgecolor(), transparent = poster)
    return

def hist_direction(df, cid, summary, size, saveto, poster, details):
    # fig, axs = plt.subplots(1, len(TERRAINS), figsize = size, linewidth = 5*poster*neatline, edgecolor = 'k')
    # for i, tc in enumerate(TERRAINS):
    #     dft = df[df['terrain'] == tc]
    #     axs[i].hist(dft['wd_2'], bins = 1000)
    # fig.tight_layout()

    #dfspec = df[(df[f'wd_2'] > 322) & (df[f'wd_2'] < 323)]
    dfspec = df[df['terrain'] == 'complex']
    plt.hist(dfspec[f'wd_2'], bins = 100)
    plt.title('10 meter wind direction in complex terrain')
    plt.xlabel('wind direction, degrees')
    plt.ylabel('number of records')
    plt.savefig(saveto)
    return

def tmp_plot_directions(df, *args, **kwargs):
    start = '2018-02-18 12:00:00'
    end = '2018-02-24 12:00:00'
    dfr = df[df['time'].between(start, end, inclusive='both')]
    fig, axs = plt.subplots(6)
    for i in range(6):
        axs[i].scatter(dfr['time'], dfr[f'wd_{i+1}'], s = 0.1)
    plt.show()

ALL_PLOTS = {
    'bar_stability': ('Stability Frequency Bar Plot', bar_stability, (4,3)),
    'annual_profiles' : ('Annual Wind Profiles with Fits, by Terrain', annual_profiles, (6.5,3)),
    'wse_histograms' : ('Histograms of WSE, by Stability, including Terrain', wse_histograms, (6.5,4.5)),
    'veer_histograms' : ('Histograms of Wind Veer, by Stability', veer_histograms, (6.5,4.5)),
    'veer_profiles' : ('Wind Direction Profiles, by Terrain', veer_profiles, (6.5,3)),
    'tod_wse' : ('Time of Day Plots of WSE, by Terrain, including Stability & Fits', tod_wse, (6.5,6)),
    'data_gaps' : ('Data Gap Visualization', data_gaps, (8,4)),
    'terrain_breakdown' : ('Breakdown of Terrain Characterizations, by Month', terrain_breakdown, (6.5,4.5)),
    'windrose_comparison' : ('KCC/CID Wind Rose Comparison', windrose_comparison, (6.5,3)),
    'pti_profiles' : ('Annual Pseudo-TI Profiles, by Terrain', pti_profiles, (6.5,3)),
    'speed_distributions' : ('Distributions of Wind Speeds, by Height', speed_distributions, (6.5,6)),
    'pti_distributions' : ('Distributions of Pseudo-TI, by Height', pti_distributions, (6.5,6)),
    'pti_vs_wse' : ('106 meter Pseudo-TI vs WSE', pti_vs_wse, (6.5,3)),
    'pti_vs_rib' : ('106 meter Pseudo-TI vs Bulk Ri', pti_vs_rib, (6.5, 3)),
    'pti_vs_drms' : (f'{int(HEIGHTS_GAPS[DRMSBOOM-1])} meter Pseudo-TI vs Directional RMS', pti_vs_drms, (6.5, 3)),
    'correlations' : ('Correlation Coefficients', correlations, (7,7)),
    'determinations' : ('Determination Coefficients', determinations, (7,7)),
    #'hist_direction' : ('Wind direction distributions', hist_direction, (6.5,9)),# TMP
    #'plt_direction' : ('Feb 20-22 Wind directions', tmp_plot_directions, (6.5,9))#TMP
}

def list_possible_plots():
    print('Possible plots to generate:')
    for tag in ALL_PLOTS.keys():
        print(f'\t{tag}')

JUSTONE = None

def generate_plots(df: pd.DataFrame, cid: pd.DataFrame, savedir: str, summary: dict, which: list = [JUSTONE] if JUSTONE is not None else ALL_PLOTS.keys(), poster: bool = False, details: bool = False, **kwargs):
    if details:
        print('details = True (additional details will be printed)')
        for sc in df['stability'].unique():
            print(f"In stability class '{sc}':")
            dfs = df[df['stability'] == sc]
            N = len(dfs[dfs['alpha'] < 0])
            print(f'\t{N} have alpha < 0 ({100*N/len(dfs):.3f}%)')
        for b in BOOMS:
            print(f"At height {h} meters:")
            print(f'\tMean gust factor is {df[f"gust_{b}"].mean():.4f}')
            print(f'\tMedian gust factor is {df[f"gust_{b}"].median():.4f}')
        #alpha_corrs(df) # UNCOMMENT FOR MORE DETAILS ON ALPHA'S CORRELATIONS WITH TRANSFORMED QTYS
    plt.rcParams['font.size'] = 13 if poster else 14
    plt.rcParams['font.family'] = 'sans-serif' if poster else 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans' if poster else 'stix'
    print(f'Generating final plots in {"Poster" if poster else "Paper"} mode')
    if not details: print('Details suppressed. Rerun with details = True (-v in kcc.py) to print.')
    not_generated = list(ALL_PLOTS.keys())
    fig_savedir = f'{savedir}/{"P" if poster else ""}{summary["_rules_chksum"]}'
    os.makedirs(fig_savedir, exist_ok = True)
    for tag in which:
        long, plotter, size = ALL_PLOTS[tag]
        print(f"Generating plot {tag}: '{long}'")
        size = (2*size[0],2*size[1])
        plotter(df = df, cid = cid, summary = summary, size = size, saveto = f'{fig_savedir}/{tag}.png', poster = poster, details = details)
        not_generated.remove(tag)
    if len(not_generated) != 0:
        print(f'Plots not generated: {not_generated}') 
    print(f'Finished generating plots. Final plots saved to:\n\t{fig_savedir}/')
    print('Rules used in analysis to create plot data:')
    for key, val in summary.items():
        if key[0] != '_':
            print(f'\t{key} = {val}')
