#!/usr/bin/env python3


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

    xstart = np.where(x_list > 15.)[0][0]
    xend = np.where(x_list < 18.)[0][-1]

    print(xstart, xend)
    gradient_check = (np.log(y_list[xend]) - np.log(y_list[xstart])) / (x_list[xend] - x_list[xstart])
    intercept = np.log(y_list[xstart]) - gradient_check * x_list[xstart]
    print(gradient_check)

    xline = x_list[xstart: xend]
    yline = xline * gradient_check + intercept
    plt.plot(xline, yline, 'r-')
    plt.scatter(x_list, np.log(y_list))
    plt.title("Number Count Plot",fontsize=14)
    plt.xlabel("Magnitude", fontsize=14)
    plt.ylabel("log(N)",fontsize=14)
    # plt.yscale("log")
    plt.savefig("output/numberCounts.png")
    plt.show()
