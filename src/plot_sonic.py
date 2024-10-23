import pandas as pd
import matplotlib.pyplot as plt
import helper_functions as hf
import numpy as np

UNITS = {
    'mean_u' : 'm/s',
    'mean_v' : 'm/s',
    'mean_w' : 'm/s',
    'slow_mean_u' : 'm/s',
    'slow_mean_v' : 'm/s',
    'lapse_mean' : 'K/m',
    'lapse_median' : 'K/m',
    'rms' : 'm/s',
    'slow_rms' : 'm/s',
    'tke' : 'J/kg',
    'L' : 'm',
    'ustar' : 'm/s',
    'length_scale' : 'm',
    'delta_dir' : 'deg'
}

"""
PLOTNAMES = {
    "autocorr" : ("autocorrs.png", "aligned_autocorrs.png"),
    "data" : ("data.png", "aligned_data.png"),
    "fluxes" : ("fluxes.png", "fluxes.png")
}

SONIC_DIRECTORY = "../outputs/sonic_sample"
"""

class Frame:
    def __init__(self, location = None, dataframe = None):
        if location is not None and dataframe is None:
            df = pd.read_csv(location)
            df['start'] = pd.to_datetime(df['start'])
            df['end'] = pd.to_datetime(df['end'])
            df.set_index('start', inplace = True)
            df.sort_index(inplace = True)
            self.df = df
        elif dataframe is not None:
            self.df = dataframe
        else:
            print('Failed to initialize frame')

    def compare(self, first, second, fit=False):
        if first in UNITS.keys():
            plt.xlabel(f'{first} ({UNITS[first]})')
        else:
            plt.xlabel(first)
        if second in UNITS.keys():
            plt.ylabel(f'{second} ({UNITS[second]})')
        else:
            plt.ylabel(second)
        plt.scatter(self.df[first], self.df[second])
        if fit:
            A, B = hf.ls_linear_fit(self.df[first], self.df[second])
            linex = [np.min(self.df[first]), np.max(self.df[first])]
            liney = [A + B*x for x in linex]
            plt.plot(linex, liney, label=f'{A:.3f} + {B:.3f} * x')
            plt.legend()
        plt.show()
        return
    
    def ccompare(self, first, second, flag = 'sflag', max = 0, zeroax = False, fit = False, id = False, title = None):
        ins = self.df[self.df[flag] <= max]
        outs = self.df[self.df[flag] > max]
        if first in UNITS.keys():
            plt.xlabel(f'{first} ({UNITS[first]})')
        else:
            plt.xlabel(first)
        if second in UNITS.keys():
            plt.ylabel(f'{second} ({UNITS[second]})')
        else:
            plt.ylabel(second)
        plt.grid(which='both')
        plt.scatter(ins[first], ins[second], c='b', label='High quality')
        plt.scatter(outs[first], outs[second], c='r', label='Low quality')
        if fit or id:
            linex = [np.min(self.df[first]), np.max(self.df[first])]
        if fit:
            A_all, B_all = hf.ls_linear_fit(self.df[first], self.df[second])
            liney_all = [A_all + B_all*x for x in linex]
            plt.plot(linex, liney_all, label=f'All: {A_all:.3f} + {B_all:.3f} * x', c='green', linestyle='dashed')
            A_ins, B_ins = hf.ls_linear_fit(ins[first], ins[second])
            liney_ins = [A_ins + B_ins*x for x in linex]
            plt.plot(linex, liney_ins, label=f'Good: {A_ins:.3f} + {B_ins:.3f} * x', c='blue', linestyle='dashed')
        if id:
            plt.plot(linex, linex, label='Identity', c='purple', linestyle='dashed')
        if title is not None and type(title) is str:
            plt.title(title)
        plt.legend()
        plt.show()
        return
    
    def timeplot(self, variable):
        plt.xlabel('datetime')
        if variable in UNITS.keys():
            plt.ylabel(f'{variable} ({UNITS[variable]})')
        else:
            plt.ylabel(variable)
        plt.scatter(self.df.index, self.df[variable])
        plt.show()
        return()
    
    def cut(self, variable, lower = None, upper = None):
        cutdf = self.df.copy()
        if lower is not None:
            cutdf = cutdf[cutdf[variable] >= lower]
        if upper is not None:
            cutdf = cutdf[cutdf[variable] <= upper]
        return Frame(dataframe = cutdf)

    def signtable(self, first, second):
        total = len(self.df)
        firstneg = self.df[self.df[first] < 0]
        firstpos = self.df[self.df[first] > 0]
        negneg = len(firstneg[firstneg[second] < 0])
        negpos = len(firstneg[firstneg[second] > 0])
        posneg = len(firstpos[firstpos[second] < 0])
        pospos = len(firstpos[firstpos[second] > 0])
        print(f'For {first} and {second}:')
        print(f'-/-: {negneg} ({(100*negneg/total):.1f} %)')
        print(f'-/+: {negpos} ({(100*negpos/total):.1f} %)')
        print(f'+/-: {posneg} ({(100*posneg/total):.1f} %)')
        print(f'+/+: {pospos} ({(100*pospos/total):.1f} %)')
    
    def print(self):
        print(self.df)
        return
    
def compareLtoRi(df):
    df.compare('L','Rib_mean')
    df.compare('L','Rif')
    df.signtable('L','Rib_mean')
    df.signtable('L','Rif')

if __name__ == '__main__':
    df = Frame('../outputs/sonic_sample/summary.csv')
    #df.compare('Rif','wu')
    #df.ccompare('Rif','wu')
    df.ccompare('mean_u','slow_mean_u',id=True,title='U wind speeds with identity for comparison')
    df.ccompare('mean_v','slow_mean_v',id=True,title='V wind speeds with identity for comparison')
    df.ccompare('mean_u','slow_mean_u',fit=True,title='U wind speeds with best fit')
    df.ccompare('mean_v','slow_mean_v',fit=True,title='V wind speeds with best fit')
    #df.ccompare('Rif','Rib_median')
    #compareLtoRi(df)
    #df.compare('instationarity','itc_dev',fit=True)
    #compareLtoRi(df)
    #dfc = df.cut('sflag', upper = 0)
    #compareLtoRi(dfc)
