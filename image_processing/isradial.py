import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors

from numba import njit

def normalTest(testData, threshold):
    """
    returns true or false
        depending on whether testData has normal distribution

    note: inaccurate if ...
    """
    # for this range use normaltest
    if 20 < len(testData) < 50:
        print("use normaltest:")
        p_value = stats.normaltest(testData)[1]

    ## shapiro-wilks method for small samples
    elif len(testData) < 50:
        print("use shapiro:")
        p_value= stats.shapiro(testData)[1]

    ###lillifor method
    elif len(testData) >= 50:
        print("use lillifors:")
        p_value= lilliefors(testData)[1]

    if p_value < threshold:
        print("data is not normal distributed")
        return False

    else:
        print("data is normal distributed")
        return True

@njit
def quartile_test(testData, threshold):

    quart_index = round(len(testData)/4)
    flat_value = np.sum(testData[:quart_index]) + np.sum(testData[-1:-quart_index])
    peak_value = np.sum(testData[quart_index: quart_index*3])
    if flat_value/peak_value < threshold:
        return True

    else:
        return False


## cummulative vertical and horizontal height band
@njit
def test_band(image, centre, radius, band_radius):

    xlen = len(image[0])
    ylen = len(image)
    x0 = centre[0] - radius
    x1 = centre[0] + radius
    y0 = centre[1] - radius
    y1 = centre[1] + radius
    if x0 > 0 and x1 < xlen and y0 > 0 and y1 < ylen:
        norm_test1 = image[centre[1], x0 : x1]
        norm_test2 = image[y0 : y1, centre[0]]

        for index in range(1, band_radius +1):

            norm_test1 = np.add(norm_test1, image[centre[1] + index, x0 : x1])
            norm_test1 = np.add(norm_test1, image[centre[1] - index, x0 : x1])

        for index in range(1, band_radius +1):
            norm_test2 = np.add(norm_test2, image[y0 : y1, centre[0] + index])
            norm_test2 = np.add(norm_test2, image[y0 : y1, centre[0] - index])

    else:
        norm_test1 = np.array([0.])
        norm_test2 = np.array([0.])

    return norm_test1, norm_test2
