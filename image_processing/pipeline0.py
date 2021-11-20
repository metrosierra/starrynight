#!/usr/bin/env python3
#%%%%%%%%%%%%%%%%%%%%
from astropy.io import fits
import numpy as np
import sys
from matplotlib import image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import cv2
from numba import njit

import stat_methods as stat
import masks
import photometry as phot
import sharpen as sharp

master = fits.open('../../A1_mosaic.fits')
metadata = master[0].header
# print(metadata)
metatxt = open('metadata.txt', 'w')
metatxt.write(str(metadata))


### discrete pixel values, 16 bit
def make_hist(image, bins = 2**16):

    dummy = np.copy(image)
    dummy = dummy.flatten()
    dummy = np.sort(dummy)
    width = 1.
    counts, edges, stuff = plt.hist(dummy, bins, histtype='step')

    return counts, edges, width



pixel_data = master[0].data
#%%%%%%%%%%%%%%%%%%%%%%

mask1 = masks.del_grays(pixel_data, 3421)
pixel1 = np.multiply(mask1, pixel_data)

histy, edges, width = make_hist(pixel1, 2**16)
histx = edges + width/2.
# ### fit skewed gaussian
domain = [np.where(histx > 3350)[0][0], np.where(histx < 3600)[0][-1]]
start = domain[0]
stop = domain[1]

histy_fit = histy[start: stop]
histx_fit = histx[start: stop]

peak_index = np.where(histy_fit == np.max(histy_fit))[0][0]

mean_guess = histx_fit[peak_index]
print(mean_guess, 'hi')

amp_guess = np.max(histy_fit)
print(amp_guess,' hihi')

var_guess = (np.dot(histx_fit**2, histy_fit - np.min(histy_fit)) - mean_guess**2)/(amp_guess**2)

histx_fit = histx_fit[0:peak_index + 2]
histy_fit = histy_fit[0:peak_index + 2]

fit_output = stat.fit(stat.gauss, histx_fit, histy_fit, initials = [amp_guess, mean_guess, var_guess**0.5, np.min(histy_fit)], args = None, yerr = np.sqrt(histy_fit))

#%%%%%%%%%%%%%%

xfit = np.linspace(histx_fit[0], histx_fit[-1], 1000)
yfit = stat.gauss(fit_output[0], xfit)
yfit = stat.gauss([3.41682269e+05, 3.4204370e+03, 1.42350441e+02, 1.68549141e+03]
, xfit)


aspect = 1.0
fig, ax = plt.subplots(figsize = (9,9))

# just formatting your graph
ax.tick_params(labelsize = 15, direction = 'in', length = 6, width = 1, bottom = True, top = True, left = True, right = True)
# ax.yaxis.set_major_formatter(FormatStrFormatter(yaxis_dp))
# ax.xaxis.set_major_formatter(FormatStrFormatter(xaxis_dp))


plt.grid(which = 'both')
plt.plot(xfit, yfit, linewidth = 2, color = 'r', label = '"Half-Gaussian Fit"')
plt.errorbar(
    histx[:-1],
    histy,
    yerr = histy*0.05,
    marker = '.',
    drawstyle = 'steps-mid',
    color = 'black',
    label = 'Image Histogram'
)
plt.legend(fontsize = 18)
# plt.ylim([0, np.max(histy_fit)])
plt.xlim([3300, 3600])
plt.xlabel('Pixel Value', fontsize = 18)
plt.ylabel('Bin Count', fontsize = 18)
plt.show()
