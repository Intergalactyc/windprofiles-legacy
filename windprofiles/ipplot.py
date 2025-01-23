import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.stats as spstats
import windprofiles.lib.stats as stats
import datetime
from astral import LocationInfo
from astral.sun import sun

HEIGHTS = [6,10,20,32,80,106]

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
seasons = {'Fall' : ['Sep', 'Oct', 'Nov'],
           'Winter' : ['Dec', 'Jan', 'Feb'],
           'Spring' : ['Mar', 'Apr', 'May'],
           'Summer' : ['Jun', 'Jul', 'Aug']
           }

# Solstices/equinoxes in 2018
centerdates = {
    'Fall' : datetime.date(2018, 9, 22),
    'Winter' : datetime.date(2018, 12, 21),
    'Spring' : datetime.date(2018, 3, 20),
    'Summer' : datetime.date(2018, 6, 21)
}

def time_to_hours(dt: datetime.datetime):
    return dt.hour + dt.minute / 60 + dt.second / 3600

location = LocationInfo(name = 'KCC tower', region = 'IA, USA', timezone = 'US/Central', latitude = 41.91, longitude = -91.65)

def hist_alpha_by_stability(df, separate = False, compute = True, overlay = True):
    dfc = df.copy().dropna(subset = ['stability'], axis = 0)
    uniques = list(dfc['stability'].unique())

    titleextra = ''
    if separate:
        if len(uniques) % 2 != 0:
            uniques.append(None)

        fig, axs = plt.subplots(nrows = 2, ncols = len(uniques) // 2)

        for sc, ax in zip(uniques, axs.reshape(-1)):
            if sc is None:
                ax.set_visible(False)
                continue
            df_restricted = dfc[dfc['stability'] == sc]
            label = sc.title()
            if compute or overlay:
                mean = df_restricted['alpha'].mean()
                std = df_restricted['alpha'].std()
            if compute:
                label += f': {mean:.2f}±{std:.2f}'
            ax.set_xlabel(r'$\alpha$')
            ax.set_ylabel('Probability Density')
            ax.hist(df_restricted['alpha'],
                    bins = 50,
                    density = True,
                    range = (-0.4, 1.25),
                    alpha = 0.75,
                    edgecolor = 'k',
                    )     
            if overlay:
                x = np.linspace(-0.4, 1.25, 100)
                ax.plot(x, spstats.norm.pdf(x, mean, std))
                titleextra = '\nNormal distributions overlaid'
            label += f'\nN = {len(df_restricted)}'
            ax.set_title(label)

    else:
        fig, ax = plt.subplots()
        for i, sc in enumerate(uniques):
            df_restricted = dfc[dfc['stability'] == sc]
            label = sc.title()
            if compute:
                mean = df_restricted['alpha'].mean()
                std = df_restricted['alpha'].std()
                label += f': {mean:.2f}±{std:.2f}'
            ax.hist(df_restricted['alpha'],
                    bins = 50,
                    density = True,
                    range = (-0.4, 1.25),
                    alpha = 0.55 - 0.05*i,
                    edgecolor = 'k',
                    label = label,
                    )            
        ax.legend()

    fig.suptitle(r'$\alpha$ Distribution by Stability' + titleextra)

    plt.tight_layout()
    plt.show()
    return

def alpha_tod_violins(df, season = None, local = True, wrap0 = True, fit = False, saveto = None): 

    timing = 'time'
    timezone = 'local' if local else 'UTC'
    
    if season is None:
        dfS = df.copy()
        s_text = 'full year'
    else:
        mons = seasons[season.title()]
        monnums = [months.index(m)+1 for m in mons]
        dfS = df[df[timing].dt.month.isin(monnums)].copy()
        s_text = season

    fig, ax = plt.subplots(figsize = (8, 6) if season is None else (10, 6))

    dataset = [dfS[dfS[timing].dt.hour == hr]['alpha'].reset_index(drop=True) for hr in range(24)]
    if wrap0: dataset.append(df[df[timing].dt.hour == 0]['alpha'].reset_index(drop=True))    

    ax.violinplot(dataset,
                   positions = range(25) if wrap0 else range(24),
                   showextrema = False,
                   showmedians = True,
                   widths = 0.8,
                   points = 200,
                   )

    if fit:
        medians = [dat.median() for dat in dataset]
        stds = [dat.std() for dat in dataset]
        fitsine, params = stats.fit_sine(range(24), medians[:24], stds[:24], fix_period=True)
        print(f'{s_text} alpha = {params[0]:.4f} * sin({params[1]:.4f} * t + {params[2]:.4f}) + {params[3]:.4f}')
        # NOTE: MIGHT BE NICE TO MAKE THAT A REAL PHASE SHIFT RATHER THAN NORMALIZED
        xplot = np.linspace(0, 24, 100)
        ax.plot(xplot, fitsine(xplot), color = 'red', linestyle = 'dashed', alpha = 0.5)

    major_tick_locations = range(0,25,6) if wrap0 else range(0,24,6)
    major_tick_labels = [6*i for i in range(4)]
    if wrap0: major_tick_labels.append(0)
    ax.set_xticks(ticks = major_tick_locations,
               labels = major_tick_labels,
               minor = False) # Major x ticks
    ax.set_xticks(range(24), range(24), minor = True, size=7) # Minor x ticks
    ax.set_xlabel(f'Hour into day ({timezone})')

    ax.set_ylim(-0.4,1.2)
    ax.set_ylabel(r'$\alpha$')

    if season is not None:
        s = sun(location.observer, date = centerdates[season.title()], tzinfo = location.timezone)
        ax.vlines([time_to_hours(s['sunrise']), time_to_hours(s['sunset'])], linestyle = 'dashed', ymin = -0.4, ymax = 1.2, color = 'green')

    fig.suptitle(f'WSE Medians and Distributions by Time of Day ({s_text})')

    fig.tight_layout()

    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto, bbox_inches='tight')

def alpha_tod_violins_by_terrain(df, season = None, local = True, wrap0 = True, fit = False, saveto = None):  
    # need to modify to add seasonality - currently basically identical to above
    
    timing = 'time'
    timezone = 'local' if local else 'UTC'

    if season is None:
        dfS = df.copy()
        s_text = 'full year'
    else:
        mons = seasons[season.title()]
        monnums = [months.index(m)+1 for m in mons]
        dfS = df[df[timing].dt.month.isin(monnums)].copy()
        s_text = season
    
    fig, ax = plt.subplots(figsize = (8, 6) if season is None else (10, 6))

    colors = {'open' : '#ff7f0e', 'complex' : '#1f77b4'}

    for tc in ['open', 'complex']:
        
        dfT = dfS[dfS['terrain'] == tc]
        dataset = [dfT[dfT[timing].dt.hour == hr]['alpha'].reset_index(drop=True) for hr in range(24)]
        if wrap0: dataset.append(dfT[dfT[timing].dt.hour == 0]['alpha'].reset_index(drop=True))

        parts = ax.violinplot(dataset,
                       positions = range(25) if wrap0 else range(24),
                       showextrema = False,
                       showmedians = True,
                       widths = 0.8,
                       points = 200,
                       )
        
        for pc in parts['bodies']:
            pc.set_facecolor(colors[tc])
            pc.set_alpha(0.2)
        parts['cmedians'].set_edgecolor(colors[tc])

        if fit:
            medians = [dat.median() for dat in dataset]
            stds = [dat.std() for dat in dataset]
            fitsine, params = stats.fit_sine(range(24), medians[:24], stds[:24], fix_period=True)
            print(f'{s_text} {tc} alpha = {params[0]:.4f} * sin({params[1]:.4f} * t + {params[2]:.4f}) + {params[3]:.4f}')
            # NOTE: MIGHT BE NICE TO MAKE THAT A REAL PHASE SHIFT RATHER THAN NORMALIZED
            xplot = np.linspace(0, 24, 100)
            ax.plot(xplot, fitsine(xplot), color = colors[tc], linestyle = 'dashed', alpha = 0.5)
    
    if season is not None:
        s = sun(location.observer, date = centerdates[season.title()], tzinfo = location.timezone)
        ax.vlines([time_to_hours(s['sunrise']), time_to_hours(s['sunset'])], linestyle = 'dashed', ymin = -0.4, ymax = 1.2, color = 'green')

    major_tick_locations = range(0,25,6) if wrap0 else range(0,24,6)
    major_tick_labels = [6*i for i in range(4)]
    if wrap0: major_tick_labels.append(0)
    ax.set_xticks(ticks = major_tick_locations,
               labels = major_tick_labels,
               minor = False) # Major x ticks
    ax.set_xticks(range(24), range(24), minor = True, size=7) # Minor x ticks
    ax.set_xlabel(f'Hour into day ({timezone})')

    ax.set_ylim(-0.4,1.2)
    ax.set_ylabel(r'$\alpha$')

    fig.suptitle(f'WSE Median and Distribution by Time of Day ({s_text})')

    labels = [(mpatches.Patch(color=color), tc) for tc, color in colors.items()]
    ax.legend(*zip(*labels), loc=2)

    fig.tight_layout()

    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto, bbox_inches='tight')

def ri_tod_violins(df, season = None, local = True, wrap0 = True, fit = False, cut = 20, printcutfrac = False, bounds = (-3,3)): 

    timing = 'time'
    timezone = 'local' if local else 'UTC'
    
    if season is None:
        dfS = df.copy()
        s_text = 'full year'
    else:
        mons = seasons[season.title()]
        monnums = [months.index(m)+1 for m in mons]
        dfS = df[df[timing].dt.month.isin(monnums)].copy()
        s_text = season

    dataset = [dfS[dfS[timing].dt.hour == hr]['Ri_bulk'].reset_index(drop=True) for hr in range(24)]
    if wrap0: dataset.append(df[df[timing].dt.hour == 0]['Ri_bulk'].reset_index(drop=True))

    pre_totals = [len(hourset) for hourset in dataset]
    dataset = [hourset[np.abs(hourset) < cut] for hourset in dataset]
    post_totals = [len(hourset) for hourset in dataset]
    missings = [pre-post for pre, post in zip(pre_totals, post_totals)]
    frac_missings = [mis/pre for mis, pre in zip(missings, pre_totals)]
    print(frac_missings)

    plt.violinplot(dataset,
                   positions = range(25) if wrap0 else range(24),
                   showextrema = False,
                   showmedians = True,
                   widths = 0.8,
                   points = 1000,
                   )

    if fit:
        medians = [dat.median() for dat in dataset]
        stds = [dat.std() for dat in dataset]
        fitsine, params = stats.fit_sine(range(24), medians[:24], stds[:24], fix_period=True)
        print(f'alpha = {params[0]:.4f} * sin({params[1]:.4f} * t + {params[2]:.4f}) + {params[3]:.4f}')
        # NOTE: MIGHT BE NICE TO MAKE THAT A REAL PHASE SHIFT RATHER THAN NORMALIZED
        xplot = np.linspace(0, 24, 100)
        plt.plot(xplot, fitsine(xplot), color = 'red', linestyle = 'dashed', alpha = 0.5)

    major_tick_locations = range(0,25,6) if wrap0 else range(0,24,6)
    major_tick_labels = [6*i for i in range(4)]
    if wrap0: major_tick_labels.append(0)
    plt.xticks(ticks = major_tick_locations,
               labels = major_tick_labels,
               minor = False) # Major x ticks
    plt.xticks(range(24), range(24), minor = True, size=7) # Minor x ticks
    plt.xlabel(f'Hour into day ({timezone})')

    plt.ylim(*bounds)
    plt.ylabel(r'$Ri_{b}$')

    plt.title(f'Ri_bulk Medians and Distributions by Time of Day ({s_text})')

    plt.tight_layout()
    plt.show()

def alpha_over_time(df: pd.DataFrame):
    plt.scatter(df['time'], df['alpha'], s = 1)
    plt.xlabel(f'Time')
    plt.ylabel(r'$\alpha$')
    plt.title(r'WSE $\alpha$ over time')
    plt.tight_layout()
    plt.show()

def comparison(df: pd.DataFrame, which: list[str], id = False, xlims = None, ylims = None):
    a, b = which
    plt.scatter(df[a], df[b], s=0.2)
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    plt.title(f'Comparison of {a} vs {b}')
    plt.xlabel(a)
    plt.ylabel(b)
    plt.show()

"""
def boom_data_available(df, heights, *, freq = '10min'):
    print('a')
    alltimes = pd.date_range(df['time'].min(), df['time'].max(), freq=freq).to_series()
    print('b')
    for i, height in enumerate(heights):
        print(i)
        availableData = df.apply(lambda row : height * int(not pd.isna(row[f'ws_{height}m'])), axis = 1)
        unavailableData = availableData.apply(lambda row : height - row)
        availableData[availableData == 0] = np.nan
        unavailableData[unavailableData == 0] = np.nan
        if i == 0:
            plt.scatter(df['time'], availableData, s=4, c='blue', label = 'available')
            plt.scatter(df['time'], unavailableData, s=4, c='red', label = 'unavailable')
        else:
            plt.scatter(df['time'], availableData, s=4, c='blue')
            plt.scatter(df['time'], unavailableData, s=4, c='red')
    print('c')
    print(np.array(alltimes.values))
    print(np.array(df['time']))
    fullgaps = alltimes.apply(lambda row : int(row.value not in df['time']))
    print('d')
    fullgaps[fullgaps == 0] = np.nan
    plt.scatter(alltimes, fullgaps, s=4, c='green', label = 'nowhere available')
    plt.title('Data availability/gaps')
    plt.xlabel('Time')
    plt.ylabel('Boom height (m)')
    plt.legend()
    plt.show()
    return
"""
    
# TIME HAS TO BE PASSED IN AS LOCAL TIME AND THEN SPECIFY LOCAL=TRUE ATM
