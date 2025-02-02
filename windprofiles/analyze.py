import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import os

def save(object, saveto, **kwargs): # UNTESTED; may replace with either separate versions for different objects or just an intermediate-directory-creator
    """
    Save a file, creating any intermediate directories if necessary.
    """
    os.makedirs(os.path.dirname(saveto), exist_ok = True)
    if type(object) is pd.DataFrame:
        object.to_csv(saveto)
    elif type(object) is Figure:
        object.savefig(saveto)
    elif object == 'PLOT':
        plt.savefig(saveto)
    else:
        raise Exception('analyze.save: Unrecognized object type to save')
