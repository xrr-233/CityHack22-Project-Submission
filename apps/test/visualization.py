import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot(tag, period):
    delta_x = MultipleLocator(period)
    ax = plt.gca()
    ax.xaxis.set_major_locator(delta_x)
    plt.plot(csv[0:1500]['DateTime'].values, csv[0:1500]['Y'].values)
    plt.show()

if(__name__=="__main__"):
    #csv = pd.read_csv('../../static/CH22_Demand_raw_X_Test.csv', sep=',', usecols=[0, 1, 2, 3, 4])
    csv = pd.read_csv('../../static/CH22_Demand_XY_Train.csv', sep=',', usecols=[0, 1, 2, 3, 4, 5])

    #end = len(csv)
    end = 24 * 7

    plot('X1', end)