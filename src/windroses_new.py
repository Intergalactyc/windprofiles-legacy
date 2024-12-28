import roses
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from tqdm import tqdm

# Default bins
DEFAULT_BINS = [0,0.89,2.24,3.13,4.47,6.71,8.94]
# Heights at which data is found, ignoring 80m because we want a good year-round picture
HEIGHTS = [6,10,20,32,106]

height = 106
df10 = pd.read_csv('../../outputs/slow/ten_minutes_labeled.csv') # 10-minute averaged data, with calculations and labeling performed by reduce.py
df10['time'] = pd.to_datetime(df10['time'])
winddata = (
        df10[['time',f'ws_{height}m',f'wd_{height}m']]
        .rename(columns = {f'ws_{height}m' : 'ws', f'wd_{height}m' : 'wd'})
    )

def animate():
    start_date = df10.loc[0,'time']
    end_date = df10.loc[len(df10)-1,'time']
    dates = pd.date_range(
        start = start_date,
        end = end_date
        )

    artists = []
    fig = plt.figure()
    fig.suptitle('106 meter wind data wind rose animation')
    #print(winddata['time'].dt.date)
    for date in tqdm(dates):
        data_date = winddata[winddata['time'].dt.date == date.date()]
        if len(data_date) < 10:
            continue
        artist = roses.windrose(fig, data_date)
        artist.set_title(date.date())
        artists.append([artist])
    wr_anim = anim.ArtistAnimation(fig = fig, artists = artists, interval = 200)
    #ffmpeg_path = 'C:/ffmpeg/bin/ffmpeg.exe'
    #writervideo = anim.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800, codec="libx264")
    #writervideo.exec_command = lambda *args: subprocess.Popen([ffmpeg_path] + list(args))
    #wr_anim.save('../../outputs/animationTEST.mp4', writer=writervideo)
    plt.show()

def plot():
    fig = plt.figure()
    rose = roses.windrose(fig, winddata)
    plt.show()

plot() # issue of rmax being too big - chose a consistent 100% so animation was good but this looks like crap now


# ADD A WAY OF SHOWING THE TERRAIN DELIMITATIONS
# WANT SEASONAL WIND ROSES WITH THIS IN PLACE!
