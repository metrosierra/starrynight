import numpy as np
import matplotlib.pyplot as plt

def mag_plot(mag_list, num):
    """
    returns a plot of log(number count) against magnitude

    input
        filepath - filepath to the output data
        stepsize - must be an integer
    """

    y_list = []  #y-axis  # number counts


    mag_list = np.array(mag_list)

    x_list = np.linspace(min(mag_list), max(mag_list), num)
    for i in range(num):

        y_list.append(np.count_nonzero(mag_list <= x_list[i]))

    y_list = np.array(y_list)

    plt.scatter(x_list, y_list)
    plt.title("Number Count Plot",fontsize=14)
    plt.xlabel("Magnitude", fontsize=14)
    plt.ylabel("log(N)",fontsize=14)
    plt.yscale("log")
    plt.savefig("/Users/jiayangzhang/Documents/Imperial/labs/year3/starrynight/image_processing/output/numberCounts.png")
    plt.show()
