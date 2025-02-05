# Currently non functional

import matplotlib.pyplot as plt

# iteration: monthly/seasonally, stability, terrain
# x axis: time, TOD, month, variable
# y axis: variable(s)
# labeling: x axis, y axis, title
# allow other args
# legend?
# colors?
# local time or UTC?
# overlay: storms

TERRAIN_COLORS = {
    'open' : 
    'complex' : 
}

class PlotIterator:
    def __init__(self,
                 type = None,
                 which = None,
                 ):
        # type may be "months", "seasons", "stability", "terrain"
        if type == 'months':
        elif type == 'seasons':
        elif type == 'stability':
        elif type == 'terrain':

class Plotter:
    def __init__(self, *,
                 x_type = None,
                 y_type = "line", # "violins", "line", "bars"
                 x_variable = None,
                 y_variable = None,
                 iteration = None,
                 labels = {'title' : None, 'x axis' : 'x', 'y axis' : 'y'},
                 overlay = None, # None or "storms"
                 storms = None): # only needed if overlay == "storms"
        pass

    def generate_plot(self, ax: plt.Axes):
        pass

def generate_figure(plots: list[Plotter], shape: tuple[int] = None, size: tuple[float] = None, title = 'Untitled Figure', kwargs: dict = None):
    if shape is None:
        shape = (len(plots), 1)
    rows, cols = shape
    if size is None:
        size = (8 + 3 * (rows-1), 6 + 2.5 * (cols-1))
    fig, axs = plt.subplots(nrows = rows, ncols = cols, figsize = size, fig_kw = kwargs)
    counter = 0
    for i in range(rows):
        for j in range(cols):
            if cols == 1:
                ax = axs[i]
            else:
                ax = axs[i][j]
            plot = plots[counter]
            plot.generate_plot(ax)
            counter += 1

def iteratively_generate_figure(plot: Plotter, )

test = Plotter()

generate_figure(plots = [test])
