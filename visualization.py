import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

if(__name__=="__main__"):
    csv = pd.read_csv('CH22_Demand_XY_Train.csv', sep=',', usecols=[0, 1, 2, 3, 4, 5])

    delta_x = MultipleLocator(1000)
    ax = plt.gca()
    ax.xaxis.set_major_locator(delta_x)
    plt.plot(csv[0:len(csv)]['DateTime'].values, csv[0:len(csv)]['X1'].values)
    plt.show()
    delta_x = MultipleLocator(1000)
    ax = plt.gca()
    ax.xaxis.set_major_locator(delta_x)
    plt.plot(csv[0:len(csv)]['DateTime'].values, csv[0:len(csv)]['X2'].values)
    plt.show()
    delta_x = MultipleLocator(1000)
    ax = plt.gca()
    ax.xaxis.set_major_locator(delta_x)
    plt.plot(csv[0:len(csv)]['DateTime'].values, csv[0:len(csv)]['X3'].values)
    plt.show()
    delta_x = MultipleLocator(1000)
    ax = plt.gca()
    ax.xaxis.set_major_locator(delta_x)
    plt.plot(csv[0:len(csv)]['DateTime'].values, csv[0:len(csv)]['X4'].values)
    plt.show()
    delta_x = MultipleLocator(1000)
    ax = plt.gca()
    ax.xaxis.set_major_locator(delta_x)
    plt.plot(csv[0:len(csv)]['DateTime'].values, csv[0:len(csv)]['Y'].values)
    plt.show()