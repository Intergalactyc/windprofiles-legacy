### roses.py ###
# author: Elliott Walker
# last update: 11 July 2024
# description: Custom wind rose plotting functionality

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.ticker import FixedLocator

sb.set_style('ticks')

# class RosePlotter:
#     def __init__(self, figsize = (10,10)):
#         fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))
#         ax.set_theta_direction('clockwise')
#         ax.set_theta_zero_location('N')
#         self._fig = fig
#         self._ax = ax

#     def _speed_labels(self, bins, units):
#         labels = []
#         for left, right in zip(bins[:-1], bins[1:]):
#             if left == bins[0]:
#                 labels.append('calm')
#             elif np.isinf(right):
#                 labels.append(f'>{left} {units}')
#             else:
#                 labels.append(f'{left} - {right} {units}')
#         return labels
    
#     def _ri_labels(self, bins):
#         labels = []
#         for left, right in zip(bins[:-1], bins[1:]):
#             if np.isinf(left):
#                 labels.append(f'Ri<{right}')
#             elif np.isinf(right):
#                 labels.append(f'Ri>{left}')
#             else:
#                 labels.append(f'{left}<Ri<{right}')
#         return labels

#     def _convert_dir(self, directions, N=None):
#         """
#         Convert centered angles to left-edge radians
#         """
#         if N is None:
#             N = directions.shape[0]

#         barDir = directions * np.pi/180. - np.pi/N
#         barWidth = 2 * np.pi / N

#         return barDir, barWidth
    
#     def animate(self, )

"""
class WindroseArtist:
    def __init__(self, fig):
        ax = fig.add_subplot(polar = True)
        ax.set_theta_direction('clockwise')
        ax.set_theta_zero_location('N')
        self._ax = ax
    

    def getAxes(self):
        return self._ax
"""

# Create labels for wind speed ranges
def _speed_labels(bins, units):
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if left == bins[0]:
            labels.append('calm')
        elif np.isinf(right):
            labels.append(f'>{left} {units}')
        else:
            labels.append(f'{left} - {right} {units}')

    return labels

def _ri_labels(bins):
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if np.isinf(left):
            labels.append(f'Ri<{right}')
        elif np.isinf(right):
            labels.append(f'Ri>{left}')
        else:
            labels.append(f'{left}<Ri<{right}')
    
    return labels

# Convert centered angles to left-edge radians
def _convert_dir(directions, N=None):
    if N is None:
        N = directions.shape[0]

    barDir = directions * np.pi/180. - np.pi/N
    barWidth = 2 * np.pi / N

    return barDir, barWidth

def _rose_ax(fig, rosedata, bins, *, lines = dict(), palette=None):
    if palette is None:
        palette = sb.color_palette('inferno', n_colors=rosedata.shape[1])
    
    bar_dir, bar_width = _convert_dir(bins)
    
    #fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))
    ax = fig.add_subplot(polar = True)
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')

    for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
        if n == 0:
            # first column only
            ax.bar(bar_dir, rosedata[c1].values, 
                   width=bar_width,
                   color=palette[0],
                   edgecolor='none',
                   label=c1,
                   linewidth=0)

        # all other columns
        ax.bar(bar_dir, rosedata[c2].values, 
               width=bar_width, 
               bottom=rosedata.cumsum(axis=1)[c1].values,
               color=palette[n+1],
               edgecolor='none',
               label=c2,
               linewidth=0)

    ax.legend(loc=(0.75, 0.95), ncol=2)
    ax.set_rmax(100)
    ax.set_rticks([50,100], ['50%','100%'])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    #lcolors = ['red', 'blue']
    #for category, angles in lines.items():
    #    ax.vlines(angles, 0, 100)

    return ax

def windrose(fig, winddata, *, mode='speed', bin_size=20, N_bins=None, palette=None, units='m/s', lines = {'complex' : (300,330), 'open' : (120,150)}):
    total_count = winddata.shape[0]
    
    if N_bins:
        bin_size = 360 / N_bins
    if 360 % bin_size != 0 or bin_size < 0:
        raise('bin_size must divide 360.')
    
    directions = np.arange(0, 360, bin_size)
    dir_bins = np.arange(-bin_size/2, 360+2*bin_size/3, bin_size)[:-1]

    if mode.lower() == 'speed':
        bins = [-1, 0, 2, 4, 6, 8, np.inf]
        labels = _speed_labels(bins, units=units)
        calm_count = winddata.query('ws == 0').shape[0]
        rosedata = (
            winddata.assign(ws_bins = lambda df:
                            pd.cut(df['ws'], bins=bins, labels=labels, right=True)
                    )
                    .assign(wd_bins = lambda df:
                            pd.cut(df['wd'], bins=dir_bins, labels=directions, right=False)
                    )
                    .groupby(by=['ws_bins', 'wd_bins'], observed=False)
                    .size()
                    .unstack(level='ws_bins')
                    .fillna(0)
                    .assign(calm = lambda df: calm_count / df.shape[0])
                    .sort_index(axis=1)
                    .apply(lambda x: x / total_count * 100)
        )
    elif mode.lower() == 'ri':
        bins = [-np.inf, -0.1, 0.1, 0.25, np.inf]
        labels = _ri_labels(bins)
        rosedata = (
            winddata.assign(ri_bins = lambda df:
                            pd.cut(df['ri'], bins=bins, labels=labels, right=True)
                    )
                    .assign(wd_bins = lambda df:
                            pd.cut(df['wd'], bins=dir_bins, labels=dir_labels, right=False)
                    )
                    .groupby(by=['ri_bins', 'wd_bins'], observed=False)
                    .size()
                    .unstack(level='ri_bins')
                    .fillna(0)
                    .sort_index(axis=1)
                    .apply(lambda x: x / total_count * 100)
        )
    else:
        raise(f"Mode '{mode}' not found for plotting windrose.")

    return _rose_ax(fig, rosedata, directions, lines=lines, palette=palette)
