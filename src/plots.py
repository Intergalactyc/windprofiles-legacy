# primary generation of plots

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import numpy as np
import helper_functions as hf
import os

# note to self - the 4:30 ones are "Night", the 18:30 ones are "Noon"

df10 = pd.read_csv('../../outputs/slow/ten_minutes_labeled.csv') # 10-minute averaged data, with calculations and labeling performed by reduce.py
df10['time'] = pd.to_datetime(df10['time'])

# Useful list of all of the heights, in m, that data exists at
heights = [6,10,20,32,80,106]

stability_classes = [['unstable'],['neutral'],['stable'],['strongly stable']]
combined_stability_classes = [['unstable'],['neutral'],['stable','strongly stable']]

def scatter_ri():
    # Plot of Ri over time
    plt.scatter(df10['time'], df10['ri'],s=0.1)
    plt.show()
    return

def bar_stability(combine=False):
    # Bar chart of stability classifications
    stability_r_freqs = df10['stability'].value_counts(normalize=True)
    if combine:
        plt.bar(['Unstable\n(Ri<-0.1)','Neutral\n(-0.1<Ri<0.1)','Stable\n(0.1<Ri)'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable']+stability_r_freqs['strongly stable']], color=['mediumblue','deepskyblue','orange'])
    else:
        plt.bar(['Unstable\n(Ri<-0.1)','Neutral\n(-0.1<Ri<0.1)','Stable\n(0.1<Ri<0.25)','Strongly Stable\n(0.25<Ri)'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable'],stability_r_freqs['strongly stable']], color=['mediumblue','deepskyblue','orange','crimson'])
    plt.ylabel('Relative Frequency')
    plt.title('Wind Data Sorted by Bulk Ri Thermal Stability Classification')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='-', alpha=0.8)
    plt.show()
    return

"""
def bar_new(save=False):
    # Bar chart of stability classifications
    stability_r_freqs = df10['new_stability'].value_counts(normalize=True)
    plt.bar(['unstable','neutral','stable'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable']], color=['mediumblue','deepskyblue','orange'])#,'crimson'])
    plt.ylabel('relative frequency')
    plt.title('frequency of wind data sorted by new stability classification')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='-', alpha=0.8)
    if save:
        plt.savefig('plots/week3pres/newbar.png')
    else:
        plt.show()
    return
"""

def bar_directions():
    # Bar chart of direction-based terrain classifications
    dir_r_freqs = df10['terrain'].value_counts(normalize=True)
    plt.bar(dir_r_freqs.index, dir_r_freqs.values, color=['red','green','blue'])
    plt.ylabel('relative frequency')
    plt.title('frequency of wind data sorted by terrain classification')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='-', alpha=0.8)
    plt.show()
    return

def scatter_rat():
    # Plot of vpt_lapse_env over time
    plt.scatter(df10['time'], df10['vpt_lapse_env'],s=0.2)
    plt.xlabel('time')
    plt.ylabel(r'$\Delta \theta_{v}/\Delta z$, K/m')
    plt.show()
    return

def plot_temp(height=6):
    # plot of temperature at 6m over time
    plt.scatter(df10['time'], df10[f't_{height}m'],s=1)
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

def plot_alpha(d=False):
    plt.title('WSE over time, with comparison to temperature')
    plt.scatter(df10['time'],df10['alpha'],s=0.4, label = r'$\alpha$')
    if d: plt.plot(df10['time'],[1/7]*len(df10))
    plt.scatter(df10['time'],df10['t_10m']/50-4, s=0.3, label = r'$T/(50\text{ K})-4 \text{ K}$')
    plt.xlabel('time')
    plt.legend()
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

def scatter3():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df10['ri'],df10['vpt_lapse_env'],df10['alpha'])
    ax.set_xlabel('Ri')
    ax.set_xlim([-35,25])
    ax.set_ylabel(r'$\Delta \theta_{v}/\Delta z$')
    ax.set_zlabel(r'$\alpha$')
    ax.set_zlim([-0.3,1.25])
    plt.show()
    return

def stratified_speeds():
    fig, ax = plt.subplots()
    groups = df10.groupby('stability')
    for name, group in groups:
        ax.scatter(group['time'],group['ws_106m'],label=name,s=1)
    ax.legend()
    plt.show()
    return

def hist_ri(cutoff = 10, bins = 100):
    plt.title('Histogram of Bulk Ri Distribution')
    plt.hist(df10[np.abs(df10['ri'])<cutoff]['ri'],bins=bins, density=True)
    plt.xlabel('Ri_b')
    plt.ylabel('probability density')
    plt.show()
    return()

def hist_alpha_by_stability(combine = False, title = True):
    fig, ax = plt.subplots(figsize = (5.5,4))
    if title: fig.suptitle(r'$\alpha$ distribution by stability')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Probability Density')
    scs = combined_stability_classes if combine else stability_classes
    for sc in scs:
        df_restricted = df10[df10['stability'].isin(sc)]
        ax.hist(df_restricted['alpha'], bins=50, density = True, range = (-0.4, 1.25), alpha=0.5, edgecolor = 'k', label=sc[0].capitalize())
    ax.legend()
    plt.show()
    return

PLOTNAMES = {
    "autocorr" : ("autocorrs.png", "aligned_autocorrs.png"),
    "data" : ("data.png", "aligned_data.png"),
    "fluxes" : ("fluxes.png", "fluxes.png")
}

SONIC_DIRECTORY = "../outputs/sonic_sample"

def display_sonic_plots(plotname, aligned = True):

    if plotname in PLOTNAMES.keys():
        plotname = PLOTNAMES[plotname][int(aligned)]

    subdirs = []
    for entry in os.listdir(SONIC_DIRECTORY):
        fullpath = os.path.join(SONIC_DIRECTORY,entry)
        if os.path.isdir(fullpath):
            subdirs.append(fullpath)
    
    for sub in subdirs:
        imagepath = os.path.join(sub, plotname)
        try:
            img = mpimg.imread(imagepath)
            plt.imshow(img)
            plt.show()
        except:
            print(f"Failed to find/display image from file {imagepath}")

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
    availableData = df10.apply(lambda row : N-np.sum([int(pd.isna(row[f'ws_{height}m'])) for height in heights]), axis = 1)
    plt.scatter(df10['time'], availableData)
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
    #print(fullgaps[fullgaps == 1])
    plt.scatter(alltimes, fullgaps, s=4, c='green', label = 'nowhere available')
    plt.title('Data availability/gaps')
    plt.xlabel('Time')
    plt.ylabel('Boom height (m)')
    plt.legend()
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

if __name__ == '__main__':
    #stratified_speeds()
    #alpha_vs_lapse()
    #alpha_vs_ri()
    #scatter3()
    #bar_directions()
    #bar_stability()
    #stratandri()
    #hist_ri(cutoff=0.25,bins=50)
    #hist_alpha_by_stability(combine = True, title = False)
    #display_sonic_plots("data")
    #display_sonic_plots("autocorr")
    #display_sonic_plots("fluxes")
    #plot_speeds()
    #total_data_available()
    #boom_data_available()
    #plot_alpha()
    alpha_vs_timeofday()
