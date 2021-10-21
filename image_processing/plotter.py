import numpy as np
import matplotlib.pyplot as plt

def mag_plot(stepsize, filepath):
    """
    returns a plot of log(number count) against magnitude

    input
        filepath - filepath to the output data
        stepsize - must be an integer
    """
    data = np.loadtxt(filepath, skiprows=1, delimiter='\t')


    mag_list = []
    for index, row in enumerate(data):
        # !!!!!!!!!!!!!!!!!!!!!!!!CHANGE THE TWO LINES BELOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x,y = row[0], row[1]
        mag_list.append(y)


    x_list = []  #x-axis  # x-coor
    y_list = []  #y-axis  # number counts

    xmin = int(min(mag_list))
    xmax = int(max(mag_list)+stepsize)
    for i in range(xmin, xmax, stepsize):
        x_list.append(i)
        count = 0
        for j in range(len(mag_list)):
            if mag_list[j] <= i:
                count = count + 1
        y_list.append(count)


    plt.scatter(x_list, y_list)
    plt.title("Number Count Plot",fontsize=14)
    plt.xlabel("Magnitude", fontsize=14)
    plt.yscale("log")     #log plot
    plt.ylabel("log(N)",fontsize=14)
    plt.show()

mag_plot(stepsize = 10, filepath = "/Users/jiayangzhang/Documents/Imperial/labs/year3/starrynight/image_processing/output/centres.txt")
