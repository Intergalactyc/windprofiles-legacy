import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib import cm
import numpy as np
import helper_functions as hf
import os
from sun_position_calculator import SunPositionCalculator
import scipy.stats as stats

# Load data
df10 = pd.read_csv('../../outputs/slow/ten_minutes_labeled.csv') # 10-minute averaged data, with calculations and labeling performed by reduce.py
df10['time'] = pd.to_datetime(df10['time'])
df10['local_time'] = df10['time'].dt.tz_localize('UTC').dt.tz_convert('US/Central') # add a local time column

# List of all of the heights, in m, that data exists at
heights = [6,10,20,32,80,106]
zvals = np.linspace(0.,130.,400)

# Latitude and longitude
CRLATITUDE = 41.91 # KCC met tower latitude in degrees
CRLONGITUDE = -91.65 # Met tower longitude in degree

# Months
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
seasons = {'Fall' : ['Sep', 'Oct', 'Nov'],
           'Winter' : ['Dec', 'Jan', 'Feb'],
           'Spring' : ['Mar', 'Apr', 'May'],
           'Summer' : ['Jun', 'Jul', 'Aug']
           }
seasonal_colors = {'Fall' : 'orange', 'Summer' : 'red', 'Spring' : 'green', 'Winter' : 'blue'}

# Terrain classes
terrain_classes = ['complex', 'open']

# How to classify stability by default
default_stability_classes = ['unstable','neutral','stable','strongly stable']
default_stability_cutoffs = ['Ri<-0.1','-0.1<Ri<0.1','0.1<Ri<0.25','Ri>0.25']
default_stability = {'stability classes' : default_stability_classes,
                     'stability cutoffs' : default_stability_cutoffs,
                     'stability scheme' : hf.stability_class,
                     'stability parameter' : 'ri',
                     'colors' : ['red', 'lime', 'royalblue', 'midnightblue'],
                     }

# Five class scheme
five_stability_classes = ['strongly unstable','unstable','neutral','stable','strongly stable']
five_stability_cutoffs = ['Ri<-0.25','-0.25<Ri<-0.1','-0.1<Ri<0.1','0.1<Ri<0.25','Ri>0.25']
def five_stability_scheme(Ri):
    if Ri < -0.25:
        return 'strongly unstable'
    if Ri < -0.1:
        return 'unstable'
    if -0.1 <= Ri < 0.1:
        return 'neutral'
    if 0.1 <= Ri < 0.25:
        return 'stable'
    return 'strongly stable'
five_stability = {'stability classes' : five_stability_classes,
                     'stability cutoffs' : five_stability_cutoffs,
                     'stability scheme' : five_stability_scheme,
                     'stability parameter' : 'ri',
                     'colors' : ['red', 'orange', 'lime', 'royalblue', 'midnightblue'],
                     }

def mean_wind_profiles_by_terrain_and_stability(
    height = 10,
    stability = default_stability
):
    stability_classes = stability['stability classes']
    stability_parameter = stability['stability parameter']
    stability_scheme = stability['stability scheme']
    colors = stability['colors']
    for tc in terrain_classes:
        df = df10[df10[f'wd_{int(height)}m'].apply(hf.terrain_class) == tc]
        plt.xlabel('Mean Velocity (m/s)')
        plt.ylabel('Height (m)')
        plt.title(f'{tc.title()} terrain wind profile')
        for color, sc in zip(colors, stability_classes):
            df_sc = df[df[stability_parameter].apply(stability_scheme) == sc]
            means = df_sc[[f'ws_{h}m' for h in heights]].mean(axis=0)
            mult, wsc = hf.power_fit(heights, means.values, both=True)
            plt.scatter(means.values, heights, label=r'{sc}: $u(z)={a:.2f}z^{{{b:.3f}}}$'.format(sc=sc.title(),a=mult,b=wsc), color = color)
            plt.plot(mult * zvals**wsc, zvals, color = color)
            print(f'{tc}, {sc[0]}: mult = {mult:.4f}, alpha = {wsc:.4f}')
        plt.legend()
        plt.show()

def mean_wind_profiles_by_stability_only(
    stability = default_stability,
):
    stability_classes = stability['stability classes']
    stability_parameter = stability['stability parameter']
    stability_scheme = stability['stability scheme']
    colors = stability['colors']
    plt.xlabel('Mean Velocity (m/s)')
    plt.ylabel('Height (m)')
    plt.title(f'Overall mean wind profiles')
    for color, sc in zip(colors, stability_classes):
        df_sc = df10[df10[stability_parameter].apply(stability_scheme) == sc]
        height_stratified = df_sc[[f'ws_{h}m' for h in heights]]
        means = height_stratified.mean(axis = 0)
        # stds = height_stratified.std(axis = 0)
        mult, wsc = hf.power_fit(heights, means.values, both=True)
        plt.scatter(means.values, heights, label=r'{sc}: $u(z)={a:.2f}z^{{{b:.3f}}}$'.format(sc=sc.title(),a=mult,b=wsc), color = color)
        # Error bar - looks very bad
        # plt.errorbar(means.values, heights, xerr = stds.values, fmt = 'o', elinewidth = 0, capthick = 1.5, capsize = 4, label=r'{sc}: $u(z)={a:.2f}z^{{{b:.3f}}}$'.format(sc=sc.title(),a=mult,b=wsc), color = color)
        plt.plot(mult * zvals**wsc, zvals, color = color)
        print(f'{sc[0]}: mult = {mult:.4f}, alpha = {wsc:.4f}')
    plt.legend()
    plt.show()

def bar_stability(
    stability = default_stability,
):
    stability_classes = stability['stability classes']
    stability_parameter = stability['stability parameter']
    stability_scheme = stability['stability scheme']
    stability_cutoffs = stability['stability cutoffs']
    colors = stability['colors']
    N = len(stability_classes)
    if len(stability_cutoffs) != N:
        raise('Mismatch in stability setup list lengths')
    # Bar chart of stability classifications
    classifications = df10[stability_parameter].apply(stability_scheme)
    stability_r_freqs = classifications.value_counts(normalize=True)
    plt.bar([f'{stability_classes[i].title()}\n({stability_cutoffs[i]})' for i in range(N)], [stability_r_freqs[sc] for sc in stability_classes], color = colors)
    #plt.bar(['Unstable\n(Ri<-0.1)','Neutral\n(-0.1<Ri<0.1)','Stable\n(0.1<Ri<0.25)','Strongly Stable\n(0.25<Ri)'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable'],stability_r_freqs['strongly stable']], color=['mediumblue','deepskyblue','orange','crimson'])
    plt.ylabel('Relative Frequency')
    plt.title('Wind Data Sorted by Bulk Ri Thermal Stability Classification')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='-', alpha=0.8)
    plt.show()
    return

def fits_by_stability_and_month(
    stability = default_stability,
    data = df10
):
    stability_classes = stability['stability classes']
    stability_parameter = stability['stability parameter']
    stability_scheme = stability['stability scheme']
    result = pd.DataFrame(index = months, columns = [sc.title() for sc in stability_classes])
    for num, month in enumerate(months, 1):
        df_mon = data[data['time'].dt.month==num]
        for sc in stability_classes:
            dfMsc = df_mon[df_mon[stability_parameter].apply(stability_scheme) == sc]
            means = dfMsc[[f'ws_{h}m' for h in heights]].mean(axis = 0)
            _, wsc = hf.power_fit(heights, means.values, both=True)
            result.loc[month, sc.title()] = wsc
    return result

def plot_fits_summary(data,
                      cmap = 'viridis',
                      pretitle = 'All Data',
                      saveto = None,
                      minmax = (0,0.65),
                      percent = None,
                      clabel = r'$\alpha$',
                      xlabel = 'Month',
                      ylabel = 'Stability Class',
                      title = r'Wind Shear Exponent $\alpha$ by Month and Stability Classification'
    ):
    data = data.astype(np.float64).T
    # first plot
    if percent is not None: 
        percent = percent.astype(np.float64).T
    fig, ax = plt.subplots(figsize=(10,6))
    cax = ax.imshow(data.values, cmap = cmap, vmin = minmax[0], vmax = minmax[1], aspect = 'auto')
    cbar = fig.colorbar(cax)
    cbar.set_label(cabel)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            if percent is not None:
                ax.text(j, i, f'{data.values[i, j]:.3f}' + f'\n({percent.values[i, j]:.0f}%)', ha='center', va='center', color='white', weight = 'bold')
            else:
                ax.text(j, i, f'{data.values[i, j]:.3f}', ha='center', va='center', color='white', weight = 'bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(pretitle + '\n' + title)
    plt.tight_layout()
    if saveto is not None and type(saveto) is str:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()
    # second plot
    #plt.scatter()

def terrain_breakdown_monthly(
        data = df10,
        height = 10,
        radius = 15,
        other = True,
        total = True
):
    result = pd.DataFrame(index = months + total * ['Total'], columns = [tc.title() for tc in terrain_classes + ['other']])
    proportions = result.copy()
    for num, month in enumerate(months + total * ['Total'], 1):
        if month == 'Total':
            df_mon = data
        else:
            df_mon = data[data['time'].dt.month==num]
        for tc in terrain_classes + other * ['other']:
            count_tc = len(df_mon[df_mon[f'wd_{int(height)}m'].apply(lambda dir : hf.terrain_class(dir, radius = radius)) == tc])
            result.loc[month, tc.title()] = count_tc
            proportions.loc[month, tc.title()] = count_tc / len(df_mon)
        #proportions.loc[month, 'Total'] = len(df_mon)
    return result, proportions
            # dfMsc = df_mon[df_mon[stability_parameter].apply(stability_scheme) == sc]
            # means = dfMsc[[f'ws_{h}m' for h in heights]].mean(axis = 0)
            # _, wsc = hf.power_fit(heights, means.values, both=True)
            # result.loc[month, sc.title()] = wsc

def plot_terrain_monthly(
        data = df10,
        height = 10,
        radius = 15,
        other = True,
        show_totals = True,
        proportions = False,
        label_top = True
):
    if proportions:
        data, props = terrain_breakdown_monthly(data=data, height=height, radius=radius, other = True, total=False)
        last_num = pd.Series(np.zeros(len(months), dtype=int))
        last_y = last_num.copy()
        for tc in terrain_classes + other * ['Other']:
            num = data[tc.title()].reset_index(drop = True)
            y = props[tc.title()].reset_index(drop = True)
            plt.bar(months, y, bottom = last_y, label = tc.title())
            for i in range(len(months)):
                plt.text(i, last_y.iloc[i]+y.iloc[i]/2, f'{num.iloc[i]}\n({(100*y.iloc[i]):.1f}%)', ha='center', va='center')
            last_num += num
            last_y += y
            top_offset = np.max(last_y) / 150
        title_additional = ''
        if 'open' in terrain_classes and 'complex' in terrain_classes:
            for i in range(len(months)):
                plt.text(i, last_y.iloc[i] + top_offset, f'{(data['Open'].iloc[i]/data['Complex'].iloc[i]):.2f}', ha='center')
            title_additional = '\nTop value is Open:Complex ratio'
        plt.title(f'Terrain Breakdown (Proportions of Total)\nBased on {height}m wind directions, direction cone radius of {radius} degrees'+label_top*title_additional)
        plt.ylabel('Fraction')
    else:
        data = terrain_breakdown_monthly(data=data, height=height, radius=radius, other = True, total=False)[0]
        last = pd.Series(np.zeros(len(months), dtype=int))
        for tc in terrain_classes + other * ['Other']:
            num = data[tc.title()].reset_index(drop = True)
            plt.bar(months, num, bottom = last, label = tc.title())
            for i in range(len(months)):
                plt.text(i, last.iloc[i]+num.iloc[i]/2, num.iloc[i], ha='center', va='center')
            last += num
        top_offset = np.max(last) / 150
        title_additional = ''
        if other and show_totals:
            for i in range(len(months)):
                plt.text(i, last.iloc[i] + top_offset, last.iloc[i], ha='center')
            title_additional = '\nTop value is total number of data points'
        elif 'open' in terrain_classes and 'complex' in terrain_classes:
            for i in range(len(months)):
                plt.text(i, last.iloc[i] + top_offset, data['Complex'].iloc[i]/data['Open'].iloc[i], ha='center')
            title_additional = '\nTop value is Open:Complex ratio'
        plt.title(f'Terrain Breakdown\nBased on {height}m wind directions, direction cone radius of {radius} degrees'+label_top*title_additional)
        plt.ylabel('Number of Data Points')
    plt.xlabel('Month')
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_terrain_monthly(
        data = df10,
        height = 10,
        radius = 15,
        other = True,
        total = True,
):
    data = terrain_breakdown_monthly(data=data, height=height, radius=radius, other=other, total=total)
    print('Terrain breakdown (number of data points):')
    print(data[0])
    print('\n')
    print('Terrain breakdown (proportions):')
    print(data[1])

def fits_process_all(
    stability = default_stability,
    height = 10,
    cmap = 'viridis',
    saveplots = True,
    directory = '../../outputs/results/seasonality/',
):
    # all data
    fits_SM = fits_by_stability_and_month(stability = default_stability)
    fits_SM.to_csv(directory + f'fitsSM.csv', index_label='Month')
    plot_fits_summary(data = fits_SM, saveto = directory + f'fitsSM' if saveplots else None)
    # terrain classes
    results_by_tc = dict()
    for tc in terrain_classes:
        df = df10[df10[f'wd_{int(height)}m'].apply(hf.terrain_class) == tc]
        result_tc = fits_by_stability_and_month(stability = stability, data = df)
        result_tc.to_csv(directory + f'{int(height)}m/' + f'{tc}_fitsSM_{int(height)}m.csv', index_label='Month')
        results_by_tc[tc] = result_tc
        plot_fits_summary(data = result_tc, cmap = cmap, pretitle = tc.title() + f' Terrain (based on {int(height)}m directions)', saveto = directory + f'{int(height)}m/' + f'{tc}_fitsSM_{int(height)}m' if saveplots else None)
    deviation = results_by_tc['complex'] - results_by_tc['open']
    greater = results_by_tc['complex'].combine(results_by_tc['open'], np.maximum)
    percent = 100 * deviation / greater
    plot_fits_summary(data = deviation,
                      cmap = cmap,
                      pretitle = f'COMPLEX - OPEN DISCREPANCIES (based on {int(height)}m directions)',
                      saveto = directory + f'{int(height)}m/' + f'discrepancy_{int(height)}m' if saveplots else None,
                      minmax = (-0.2,0.2),
                      percent = percent)

def stability_plots(height = 10, stability = default_stability, cmap = 'viridis'):
    #mean_wind_profiles_by_stability_only(stability = stability)
    #mean_wind_profiles_by_terrain_and_stability(height = height, stability = stability)
    #bar_stability(stability = stability)
    fits_process_all(stability = stability, height = height, cmap = cmap)
    terrain_by_month(height = height, cmap = cmap)
    # NEXT DO SOMETHING LIKE THE FITS_PROCESS_ALL THING BUT INSTEAD GIVING A BREAKDOWN OF HOW MUCH DATA IS IN EACH TERRAIN CLASS MONTHLY
        # ** TERRAIN CLASS BREAKDOWN BY MONTH **
        # RI HISTOGRAM BY MONTH?
            # MEDIAN RI BY MONTH?
        # RI HISTOGRAM BY TERRAIN TYPE?
        # ALSO FOR FITS_PROCESS_ALL MAYBE PLOT THE DIFFERENCES COMPLEX - OPEN
            # + OPEN QUESTION OF SENSITIVITY TO TERRAIN WINDOW WIDTH (EXTEND TO 45 OR 60 DEGREES (SAME CENTER?)) AND HEIGHT USED
    
def alpha_vs_timeofday(month = None, local = True):
    timing = 'local_time' if local else 'time'
    timezone = 'local' if local else 'UTC'
    if month is not None:
        dfm = df10[df10[timing].dt.month == month].copy()
        month = 'Month ' + str(month)
    else: 
        dfm = df10.copy() # if no month is specified use full dataset
        month = 'All Data'
    dfm['secondsintoday'] = (dfm[timing].dt.hour * 60 + dfm[timing].dt.minute) * 60 + dfm[timing].dt.second 
    uniquetimes = dfm['secondsintoday'].unique()
    means = [dfm[dfm['secondsintoday'] == time]['alpha'].mean() for time in uniquetimes]
    stds = [dfm[dfm['secondsintoday'] == time]['alpha'].std() for time in uniquetimes]
    plt.title(f'WSE vs Time of Day ({month})')
    plt.errorbar(uniquetimes, means, yerr = stds, fmt = 'o', markersize=3, capsize=3, elinewidth=0.5)
    plt.xlabel(f'Time of day (seconds past midnight {timezone})')
    plt.ylabel(r'$\alpha$')
    plt.tight_layout()
    plt.show()

def alpha_vs_timeofday_with_terrain(month = None, height = 10, errorbars = False, local = True):
    timing = 'local_time' if local else 'time'
    timezone = 'local' if local else 'UTC'
    if month is not None:
        dfm = df10[df10[timing].dt.month == month].copy()
        month = 'Month ' + str(month)
    else:
        dfm = df10.copy() # if no month is specified use full dataset
        month = 'All Data'
    dfm['secondsintoday'] = (dfm[timing].dt.hour * 60 + dfm[timing].dt.minute) * 60 + dfm[timing].dt.second 
    for tc in terrain_classes:
        dft = dfm[dfm[f'wd_{int(height)}m'].apply(hf.terrain_class) == tc]
        uniquetimes = dft['secondsintoday'].unique()
        means = [dft[dft['secondsintoday'] == time]['alpha'].mean() for time in uniquetimes]
        if errorbars:
            stds = [dft[dft['secondsintoday'] == time]['alpha'].std() for time in uniquetimes]
            plt.errorbar(uniquetimes, means, yerr = stds, fmt = 'o', markersize=3, capsize=3, elinewidth=0.5, label = tc.title())
        else:
            plt.scatter(uniquetimes, means, s=4, label = tc.title())
    plt.legend()
    plt.xlabel(f'Time of day (seconds past midnight {timezone})')
    plt.ylabel(r'$\alpha$')
    plt.title(f'Mean WSE vs Time of Day, by Terrain (based on {int(height)}m directions)')
    plt.tight_layout()
    plt.show()

def alpha_vs_timeofday_with_seasons(terrain = None, height = 10, local = True):
    timing = 'local_time' if local else 'time'
    timezone = 'local' if local else 'UTC'
    if terrain in terrain_classes:
        dfT = df10[df10[f'wd_{int(height)}m'].apply(hf.terrain_class) == terrain].copy()
        tcstring = terrain.title() + f' based on {height}m'
    else:
        dfT = df10.copy()
        tcstring =  'All Data'
    for S, mons in seasons.items():
        monnums = [months.index(m)+1 for m in mons]
        dfS = dfT[dfT[timing].dt.month.isin(monnums)].copy()
        dfS['secondsintoday'] = (dfS[timing].dt.hour * 60 + dfS[timing].dt.minute) * 60 + dfS[timing].dt.second 
        uniquetimes = np.sort(dfS['secondsintoday'].unique())
        means = [dfS[dfS['secondsintoday'] == time]['alpha'].mean() for time in uniquetimes]
        plt.plot(uniquetimes, means, linewidth=1, label = S, color = seasonal_colors[S])
    plt.legend()
    plt.xlabel(f'Time of day (seconds past midnight {timezone})')
    plt.ylabel(r'$\alpha$')
    plt.title(f'Mean WSE vs Time of Day, by Season ({tcstring})')
    plt.tight_layout()
    plt.show()

def alpha_tod_violins(season = None, height = 10, local = True, wrap0 = True): 

    timing = 'local_time' if local else 'time'
    timezone = 'local' if local else 'UTC'    
    
    if season is None:
        dfS = df10
        s_text = 'full year'
    else:
        mons = seasons[season.title()]
        monnums = [months.index(m)+1 for m in mons]
        dfS = df10[df10[timing].dt.month.isin(monnums)].copy()
        s_text = season

    dataset = [dfS[dfS[timing].dt.hour == hr]['alpha'].reset_index(drop=True) for hr in range(24)]
    if wrap0: dataset.append(df10[df10[timing].dt.hour == 0]['alpha'].reset_index(drop=True))
    means = [dat.mean() for dat in dataset]
    medians = [dat.median() for dat in dataset]

    plt.violinplot(dataset,
                   positions = range(25) if wrap0 else range(24),
                   showextrema = False,
                   showmedians = True,
                   widths = 0.8,
                   points = 200,
                   )
    
    major_tick_locations = range(0,25,6) if wrap0 else range(0,24,6)
    major_tick_labels = [6*i for i in range(4)]
    if wrap0: major_tick_labels.append(0)
    plt.xticks(ticks = major_tick_locations,
               labels = major_tick_labels,
               minor = False) # Major x ticks
    plt.xticks(range(24), range(24), minor = True, size=7) # Minor x ticks
    plt.xlabel(f'Hour into day ({timezone})')

    plt.ylim(-0.4,1.2)
    plt.ylabel(r'$\alpha$')

    plt.title(f'WSE Median and Distribution by Time of Day ({s_text})')

    plt.tight_layout()
    plt.show()
# add sinusoidal fit based on medians??
# box plot overlay?

def alpha_tod_violins_by_terrain(season = None, height = 10, local = True, wrap0 = True):  
    # need to modify to add seasonality - currently basically identical to above
    
    timing = 'local_time' if local else 'time'
    timezone = 'local' if local else 'UTC'

    if season is None:
        dfS = df10
        s_text = 'full year'
    else:
        mons = seasons[season.title()]
        monnums = [months.index(m)+1 for m in mons]
        dfS = df10[df10[timing].dt.month.isin(monnums)].copy()
        s_text = season
    
    colors = {'open' : '#ff7f0e', 'complex' : '#1f77b4'}
    for tc in ['open', 'complex']:
        dfT = dfS[dfS[f'wd_{int(height)}m'].apply(hf.terrain_class) == tc]
        dataset = [dfT[dfT[timing].dt.hour == hr]['alpha'].reset_index(drop=True) for hr in range(24)]
        if wrap0: dataset.append(dfT[dfT[timing].dt.hour == 0]['alpha'].reset_index(drop=True))
        parts = plt.violinplot(dataset,
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
    
    major_tick_locations = range(0,25,6) if wrap0 else range(0,24,6)
    major_tick_labels = [6*i for i in range(4)]
    if wrap0: major_tick_labels.append(0)
    plt.xticks(ticks = major_tick_locations,
               labels = major_tick_labels,
               minor = False) # Major x ticks
    plt.xticks(range(24), range(24), minor = True, size=7) # Minor x ticks
    plt.xlabel(f'Hour into day ({timezone})')

    plt.ylim(-0.4,1.2)
    plt.ylabel(r'$\alpha$')

    plt.title(f'WSE Median and Distribution by Time of Day ({s_text})')

    labels = [(mpatches.Patch(color=color), tc) for tc, color in colors.items()]
    plt.legend(*zip(*labels), loc=2)

    plt.tight_layout()
    plt.show()

def temperature_vs_timeofday_with_seasons(height = 10, local = True):
    timing = 'local_time' if local else 'time'
    timezone = 'local' if local else 'UTC'    
    
    for S, mons in seasons.items():
        monnums = [months.index(m)+1 for m in mons]
        dfS = df10[df10[timing].dt.month.isin(monnums)].copy()
        dfS['secondsintoday'] = (dfS[timing].dt.hour * 60 + dfS[timing].dt.minute) * 60 + dfS[timing].dt.second 
        uniquetimes = np.sort(dfS['secondsintoday'].unique())
        means = [dfS[dfS['secondsintoday'] == time][f't_{int(height)}m'].mean() for time in uniquetimes]
        plt.plot(uniquetimes, means, linewidth=1, label = S, color = seasonal_colors[S])
    plt.legend()
    plt.xlabel(f'Time of day (seconds past midnight {timezone})')
    plt.ylabel('Temperature (K)')
    plt.title(f'Mean Temperature ({int(height)}m) vs Time of Day, by Season')
    plt.tight_layout()
    plt.show()

def combine_alpha_temperature_seasonality_plots(height = 10, local = True):
    timing = 'local_time' if local else 'time'
    timezone = 'local' if local else 'UTC'    
    
    for S, mons in seasons.items():
        monnums = [months.index(m)+1 for m in mons]
        dfS = df10#[df10[timing].dt.month.isin(monnums)].copy()
        dfS['secondsintoday'] = (dfS[timing].dt.hour * 60 + dfS[timing].dt.minute) * 60 + dfS[timing].dt.second 
        uniquetimes = np.sort(dfS['secondsintoday'].unique())
        meanalphas = [dfS[dfS['secondsintoday'] == time]['alpha'].mean() for time in uniquetimes]
        meanTs = [dfS[dfS['secondsintoday'] == time][f't_{int(height)}m'].mean() for time in uniquetimes]
        normalized_Ts = (meanTs - np.min(meanTs))/(2*(np.max(meanTs) - np.min(meanTs)))
        plt.plot(uniquetimes, meanalphas, linewidth=1, linestyle = 'solid', label = S + r' $\alpha$', color = seasonal_colors[S])
        plt.plot(uniquetimes, normalized_Ts, linewidth=1, linestyle = 'dashed', label = S + ' normalized temperature', color = seasonal_colors[S])
    plt.legend()
    plt.xlabel(f'Time of day (seconds past midnight {timing})')
    plt.title(f'Mean WSE vs Time of Day, by Season, with (normalized) temperature ({int(height)}m)')
    plt.tight_layout()
    plt.show()

def alpha_vs_sun_altitude():
    #sun_altitudes = df10['time'].apply(lambda t : get_position(t, CRLATITUDE, CRLONGITUDE)['altitude'])
    calculator = SunPositionCalculator()
    sun_altitudes = np.rad2deg(df10['time'].apply(lambda t : calculator.pos(t.timestamp()*1000, CRLATITUDE, CRLONGITUDE).altitude))
    # plt.scatter(df10['time'],sun_altitudes)
    plt.scatter(sun_altitudes, df10['alpha'], s=0.1)
    plt.ylim(-0.1,1.0)
    plt.show()

def consider_stratification(cutoffs = [-0.1,0.1], labels = ['unstable','neutral','stable']):
    N = len(labels)
    if N != len(cutoffs) + 1:
        print('Mismatched label/cutoff list lengths')
        return
    
    print(f'Scheme: {cutoffs}')

    total = len(df10)
    for i in range(N):
        datahere = df10.copy()
        if i < N-1:
            datahere = datahere[datahere['ri'] < cutoffs[i]]
        if i > 0:
            datahere = datahere[datahere['ri'] >= cutoffs[i-1]]
        amount = len(datahere)
        print(f'{labels[i]}: {amount} ({(100 * amount/total):.1f}%)')

def stats_ri(restriction = [5.]):
    print('OVERALL BULK RICHARDSON NUMBER STATISTICS')
    print(f'Median Ri: {np.median(df10.ri):.3f}')
    print(f'Mean Ri: {np.mean(df10.ri):.3f}')
    print(f'Std Ri: {np.std(df10.ri):.3f}')
    for r in restriction:
        dfR = df10[abs(df10.ri) < r]
        print(f'RESTRICTED (|Ri| < {r}) BULK RICHARDSON NUMBER STATISTICS')
        print(f'Restricted dataset contains {(100*len(dfR)/len(df10)):.1f}% of original data')
        print(f'Median Ri: {np.median(dfR.ri):.3f}')
        print(f'Mean Ri: {np.mean(dfR.ri):.3f}')
        print(f'Std Ri: {np.std(dfR.ri):.3f}')

def total_data_available():
    N = len(heights)
    plt.scatter(df10['time'], df10['availability'])
    plt.show()
    return

def boom_data_available():
    alltimes = pd.date_range(df10['time'].min(), df10['time'].max(), freq='10min').to_series()
    for i, height in enumerate(heights):
        availableData = df10.apply(lambda row : height * int(not pd.isna(row[f'ws_{height}m'])), axis = 1)
        unavailableData = availableData.apply(lambda row : height - row)
        availableData[availableData == 0] = np.nan
        unavailableData[unavailableData == 0] = np.nan
        if i == 0:
            plt.scatter(df10['time'], availableData, s=4, c='blue', label = 'available')
            plt.scatter(df10['time'], unavailableData, s=4, c='red', label = 'unavailable')
        else:
            plt.scatter(df10['time'], availableData, s=4, c='blue')
            plt.scatter(df10['time'], unavailableData, s=4, c='red')
    fullgaps = alltimes.apply(lambda row : int(row not in np.array(df10['time']).astype('datetime64[ns]')))
    fullgaps[fullgaps == 0] = np.nan
    plt.scatter(alltimes, fullgaps, s=4, c='green', label = 'nowhere available')
    plt.title('Data availability/gaps')
    plt.xlabel('Time')
    plt.ylabel('Boom height (m)')
    plt.legend()
    plt.show()
    return

def alpha_vs_lapse(d=False):
    df = df10.dropna(subset=['vpt_lapse_env','alpha'],how='any')
    fig, ax = plt.subplots()
    if d: ax.plot(df['vpt_lapse_env'],[1/7]*len(df))
    groups = df.groupby('stability')
    for name, group in groups:
        ax.scatter(group['vpt_lapse_env'],group['alpha'],label=name,s=0.5)
    ax.legend()
    ax.set_xlim([-0.03,0.1])
    ax.set_ylim([-0.3,1.25])
    ax.set_xlabel(r'Lapse Rate ($\Delta \theta_{v}/\Delta z$) [K/m]')
    ax.set_ylabel(r'Wind Shear Exponent ($\alpha$)')
    corr = np.corrcoef(df['vpt_lapse_env'], df['alpha'])[0,1]
    fig.suptitle(r'$r={{{r:.4f}}}$'.format(r=corr, r2=corr**2))
    plt.show()
    return

def alpha_vs_ri(d=False):
    fig, ax = plt.subplots()
    if d: ax.plot(df10['ri'],[1/7]*len(df10))
    groups = df10.groupby('stability')
    for name, group in groups:
        subgroups = group.groupby('terrain')
        for subname, subgroup in subgroups:
            if subname == 'other':
                continue
            fullname = f'{name} {subname}'
            ax.scatter(subgroup['ri'],subgroup['alpha'],label=fullname,s=3)
    ax.legend()
    ax.set_xlim([-35,25])
    ax.set_ylim([-0.3,1.25])
    ax.set_xlabel('Bulk Richardson Number (Ri)')
    ax.set_ylabel(r'Wind Shear Exponent ($\alpha$)')
    plt.show()
    return

def plot_alpha(tcolor = False, d = False, temp = None, avail = False, speed = None, title = True):
    if title:
        plt.title('WSE over time' + (temp is not None) * ', with comparison to temperature' + (speed is not None or avail) * ', and other details')
    if tcolor:
        for tc in ['open', 'complex', 'other']:
            df10_tc = df10[df10['terrain'] == tc]
            plt.scatter(df10_tc['time'], df10_tc['alpha'], s = 0.4, label = tc)
    else:
        plt.scatter(df10['time'],df10['alpha'],s=0.4, label = r'$\alpha$')
    if d: plt.plot(df10['time'],[1/7]*len(df10))
    if temp is not None:
        plt.scatter(df10['time'],df10['t_10m']/50-4, s=0.3, label = r'$T/(50\text{ K})-4$' + f'({temp} m)')
    if avail:
        plt.scatter(df10['time'], df10['availability'], label='availability', s=0.5)
    plt.gca().legend(loc='upper left')
    if speed is not None:
        ogax = plt.gca()
        twinax = ogax.twinx()
        for h in speed:
            twinax.plot(df10['time'], df10[f'ws_{h}m'], linewidth=0.2, linestyle='dashed', label=f'ws_{h}m')
        twinax.legend(loc='upper right')
        ogax.set_zorder(1)
        ogax.set_frame_on(False)
    plt.xlabel('time')
    plt.tight_layout()
    plt.show()
    return

def alpha_vs_temperature(month = None):
    if month is not None:
        dfm = df10[df10['time'].dt.month == month]
        month = 'Month ' + str(month)
        size = 0.5
    else:
        dfm = df10 # if no month is specified use full dataset
        month = 'All Data'
        size = 0.3
    plt.title(f'WSE vs Temperature at 10 meters ({month})')
    plt.scatter(dfm['t_10m'], dfm['alpha'], s=size)
    plt.xlabel('temperature (10m)')
    plt.ylabel(r'$\alpha$')
    plt.show()
    return

def plot_speeds():
    for height in heights:
        plt.scatter(df10['time'],df10[f'ws_{height}m'], label = str(height), s=1)
    plt.legend()
    plt.show()
    return

def stratandri():
    consider_stratification()
    consider_stratification(cutoffs=[-0.02,0.02])
    consider_stratification(cutoffs=[-0.17,0.02])
    consider_stratification(cutoffs=[-1,1])
    stats_ri([1,2,5,10])

def hist_ri(cutoff = 10, bins = 100):
    plt.title('Histogram of Bulk Ri Distribution')
    plt.hist(df10[np.abs(df10['ri'])<cutoff]['ri'],bins=bins, density=True)
    plt.xlabel('Ri_b')
    plt.ylabel('probability density')
    plt.show()
    return()

def hist_alpha_by_stability(classifier = hf.stability_class_3, variable = 'ri', separate = False, compute = True, overlay = True):
    dfc = df10.copy().drop(columns = ['stability'])
    dfc['stability'] = dfc.apply(lambda row : classifier(row[variable]), axis = 1)
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
                ax.plot(x, stats.norm.pdf(x, mean, std))
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

if __name__ == '__main__':
    # alpha_vs_lapse()
    # alpha_vs_ri()

    # total_data_available()
    # boom_data_available()

    # for h in [6, 10, 20]:
    #     stability_plots(stability = default_stability, height = h)
    # stability_plots(stability = five_stability)
    # alpha_vs_timeofday()

    # hist_ri(cutoff=0.25,bins=50)
    # hist_alpha_by_stability(combine = True, title = False)

    # alpha_vs_timeofday_with_terrain(height = 10)
    # alpha_vs_timeofday_with_terrain(height = 6)
    # temperature_vs_timeofday_with_seasons()
    # combine_alpha_temperature_seasonality_plots()
    # alpha_vs_sun_altitude()

    # alpha_vs_timeofday_with_seasons()
    # alpha_vs_timeofday_with_seasons(terrain = 'open')
    # alpha_vs_timeofday_with_seasons(terrain = 'complex')

    # alpha_tod_violins()
    # alpha_tod_violins_by_terrain()
    # for s in seasons.keys():
    #     alpha_tod_violins(season = s)
    #     alpha_tod_violins_by_terrain(season = s)

    hist_alpha_by_stability(classifier = hf.stability_class_3, separate = False, compute = True)
    hist_alpha_by_stability(classifier = hf.stability_class, separate = False, compute = True)

    #hist_alpha_by_stability(classifier = hf.stability_class_3, separate = True, compute = True, overlay = True)
    #hist_alpha_by_stability(classifier = hf.stability_class, separate = True, compute = True, overlay = True)

    # plot_alpha(temp=10,avail=True,speed=[6,10,20,32,106],title=False)
    # plot_alpha(tcolor=True)

    # print_terrain_monthly()
    # plot_terrain_monthly(other = False, proportions=True) # Do something like these with stability?
    # plot_terrain_monthly(other = True, proportions=False)
