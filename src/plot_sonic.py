import pandas as pd
import matplotlib.pyplot as plt

summaryfile = "../outputs/sonic_sample/summary.csv"

df = pd.read_csv(summaryfile)
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
df.set_index('start', inplace = True)
df.sort_index(inplace = True)

def printdf():
    print(df)
    return

def compare(first, second):
    plt.xlabel(first)
    plt.ylabel(second)
    plt.scatter(df[first], df[second])
    plt.show()
    return

if __name__ == '__main__':
    printdf()
    compare('Rif', 'length_scale')
