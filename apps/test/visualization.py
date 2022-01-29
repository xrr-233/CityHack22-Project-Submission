import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

if(__name__=="__main__"):
    csv = pd.read_csv('../../static/CH22_Demand_XY_Train.csv', sep=',', usecols=[0, 1, 2, 3, 4, 5])

    delta_x = MultipleLocator(24*7)
    ax = plt.gca()
    ax.xaxis.set_major_locator(delta_x)
    plt.plot(csv[0:1500]['DateTime'].values, csv[0:1500]['Y'].values)
    plt.show()