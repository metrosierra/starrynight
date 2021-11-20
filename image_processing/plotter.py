#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import stat_methods as stat



y_list = []
ratios = np.loadtxt('output/ratios.txt')
x_list = np.linspace(np.min(ratios), np.max(ratios), 30)
for i in range(30):
    y_list.append(np.count_nonzero(ratios >= x_list[i]))

y_list = np.array(y_list)

aspect = 1.0
fig, ax = plt.subplots(figsize = (9,9))

# just formatting your graph
ax.tick_params(labelsize = 15, direction = 'in', length = 6, width = 1, bottom = True, top = True, left = True, right = True)
# ax.yaxis.set_major_formatter(FormatStrFormatter(yaxis_dp))
# ax.xaxis.set_major_formatter(FormatStrFormatter(xaxis_dp))
plt.xlabel("Ellipse Axis Ratio (Major/Minor)", fontsize=18)
plt.ylabel("Count", fontsize=18)

plt.scatter(x_list, np.round(y_list * 20.5))
plt.grid(which = 'both')
plt.show()



def mag_plot(mag_list, num):
    """
    returns a plot of log(number count) against magnitude

    input
        filepath - filepath to the output data
        stepsize - must be an integer
    """

    y_list = []  #y-axis  # number counts


    mag_list = np.array(mag_list)
    x_list = np.linspace(min(mag_list), 25., num)
    for i in range(num):

        y_list.append(np.count_nonzero(mag_list <= x_list[i]))

    y_list = np.array(y_list)/ 0.046291935

    xstart = np.where(x_list > 13.5)[0][0]
    xend = np.where(x_list < 17.5)[0][-1]

    print(xstart, xend)

    xsample = x_list[xstart: xend]

    #per degree square
    ysample = y_list[xstart: xend]

    gradient_check = (np.log10(y_list[xend]) - np.log10(y_list[xstart])) / (x_list[xend] - x_list[xstart])
    intercept = np.log10(y_list[xstart]) - gradient_check * x_list[xstart]
    print(gradient_check)

    output = stat.fit(stat.linear, xsample, np.log10(ysample), initials = [gradient_check, intercept], yerr = np.log10(ysample)*np.sqrt(ysample)/ysample)

    aspect = 1.0
    fig, ax = plt.subplots(figsize = (9,9))

    # just formatting your graph
    ax.tick_params(labelsize = 15, direction = 'in', length = 6, width = 1, bottom = True, top = True, left = True, right = True)
    # ax.yaxis.set_major_formatter(FormatStrFormatter(yaxis_dp))
    # ax.xaxis.set_major_formatter(FormatStrFormatter(xaxis_dp))


    plt.grid(which = 'both')

    yline = stat.linear(output[0], xsample)
    plt.plot(xsample, yline, 'r-', label = 'ODR Fit Line')

    plt.errorbar(x_list, np.log10(y_list), yerr = np.log10(y_list) * np.sqrt(y_list)/y_list, fmt = 'o', color = 'black', mew = 1., ms = 1., capsize = 3, label = 'Number Plot')

    plt.legend(fontsize = 18)
    plt.title("Number Count Plot",fontsize=18)
    plt.xlabel("Magnitude", fontsize=18)
    plt.ylabel("log(N)/square degree",fontsize=18)
    plt.xlim([12., 22.])
    # plt.yscale("log")
    plt.savefig("output/numberCounts.png")
    plt.show()
