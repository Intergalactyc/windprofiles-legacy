import os
import pandas as pd
import numpy as np
from windrose import WindroseAxes
from calendar import month_name, month_abbr
from matplotlib import cm
import matplotlib.pyplot as plt

output_directory = '../../outputs/'
rose_out_directory = output_directory + 'windroses/'
df10_filename = output_directory + 'slow/ten_minutes_labeled.csv'
df10 = pd.read_csv(df10_filename)
df10['time'] = pd.to_datetime(df10['time'])

DEFAULT_BINS = [0,0.89,2.24,3.13,4.47,6.71,8.94]
HEIGHTS = [6,10,20,32,106]
DEFAULT_HEIGHT = 10

MONTHS = {mabbr : (i, mname) for i, mname, mabbr in zip(range(1, 13), month_name[1:], month_abbr[1:])}

SEASONS = {
    'Fall' : ['Sep', 'Oct', 'Nov'],
    'Winter' : ['Dec', 'Jan', 'Feb'],
    'Spring' : ['Mar', 'Apr', 'May'],
    'Summer' : ['Jun', 'Jul', 'Aug']
}

HALFYEARS = {
    'Primarily SE (Open)' : ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
    'Primarily NW (Complex)' : ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
}

mdMONTHS = {mabbr : [mabbr] for mabbr in MONTHS.keys()}

def cleanup(name):
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '')

def monthcut_data(month_list : list,
                  data = df10):
    
    month_num_list = [MONTHS[mabbr][0] for mabbr in month_list]

    return data[data['time'].dt.month.isin(month_num_list)]

def generate_rose(data = df10,
                  *,
                  height = DEFAULT_HEIGHT,
                  bins = DEFAULT_BINS,
                  title = None,
                  bottom_text = 'Based on data from Sept 2017 - Sept 2018',
                  saveto = None,
                  transparent = False,
                  ndata = True,
                  terrain_lines = True,
                  terrain_alpha = 0.15,
                  open = (300, 330),
                  complex = (120, 150),
                  nsector = 16,
                  ):
    
    wscol = f'ws_{height}m'
    wdcol = f'wd_{height}m'
    notnan = data[~data[wscol].isna()]
    speeds = notnan[wscol]
    directions = notnan[wdcol]

    if len(speeds) != len(directions):
        raise('Size mismatch')

    ax = WindroseAxes.from_ax()
    ax.bar(
        directions,
        speeds,
        normed = True,
        opening = 1.0,
        bins = bins,
        edgecolor = 'white',
        cmap = cm.rainbow,
        nsector = nsector
    )

    if terrain_lines:
        max_r = ax.get_ylim()[1]
        open_rad = [np.deg2rad(deg) for deg in open]
        cmpl_rad = [np.deg2rad(deg) for deg in complex]
        ax.vlines(open_rad, 0, max_r, color = 'blue', linestyle = 'dashed')
        ax.vlines(cmpl_rad, 0, max_r, color = 'red', linestyle = 'dashed')
        if terrain_alpha > 0:
            ax.fill_between(
                np.linspace(open_rad[0], open_rad[1], 100),
                0,
                max_r,
                alpha = terrain_alpha,
                color = 'blue'
            )
            ax.fill_between(
                np.linspace(cmpl_rad[0], cmpl_rad[1], 100),
                0,
                max_r,
                alpha = terrain_alpha,
                color = 'red'
            )

    ax.set_legend()

    if title is not None:
        plt.title(title)

    if bottom_text is not None:
        plt.text(
            1.04,
            -0.08,
            bottom_text,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )

    if ndata:
        N = len(speeds)
        plt.text(
            0,
            -0.08,
            f'{N=}',
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )
    
    if saveto is not None:
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        plt.savefig(saveto, bbox_inches='tight', transparent=transparent)
    else:
        plt.tight_layout()
        plt.show()

    plt.close()

    return

def monthcut_rose(month_list,
                  *,
                  data = df10,
                  height = DEFAULT_HEIGHT,
                  bins = DEFAULT_BINS,
                  bottom_text = True,
                  title = None,
                  saveto = None,
                  nsector = 16,
                  terrain_lines = True,
                  ):
        
    df_cut = monthcut_data(month_list=month_list, 
                           data=data)

    bottom_text = 'Months considered: '
    for month in month_list:
        bottom_text += month + ', '
    bottom_text = bottom_text[:-2]

    generate_rose(
        data = df_cut,
        height = height,
        bins = bins,
        title = title,
        bottom_text = bottom_text,
        saveto = saveto,
        nsector = nsector,
        terrain_lines = terrain_lines
    )

def roses_by_month_segment(segment_dict,
                         height = 10,
                         savedir = None,
                         nsector = 16,
                         terrain_lines = True,
                         ):
    
    for name, months in segment_dict.items():

        saveto = (savedir + '/'*(savedir[-1] not in ['/','\\']) + 'rose_' + cleanup(name) + '.png') if savedir is not None else None
        
        monthcut_rose(
            month_list = months,
            height = height,
            title = f'Cedar Rapids, Iowa: {name} ({height} m)',
            saveto = saveto,
            nsector = nsector,
            terrain_lines = terrain_lines
        )

if __name__ == '__main__':
    NSECTOR = 16
    roses_by_month_segment(SEASONS,
                           savedir = rose_out_directory + 'seasonal/',
                           nsector = NSECTOR,
                           terrain_lines = True,
                           )
    roses_by_month_segment(HALFYEARS,
                           savedir = rose_out_directory + 'halfyears/',
                           nsector = NSECTOR,
                           terrain_lines = True,
                           )
    roses_by_month_segment(mdMONTHS,
                           savedir = rose_out_directory + 'monthly/',
                           nsector = NSECTOR,
                           terrain_lines = False,
                           )
