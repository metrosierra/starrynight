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

import stat_methods as fitting
import masks
import photometry as phot
import sharpen as sharp
import isradial

master = fits.open('../../A1_mosaic.fits')
metadata = master[0].header
# print(metadata)
metatxt = open('metadata.txt', 'w')
metatxt.write(str(metadata))

hi = 'HAI!!!'


### discrete pixel values, 16 bit
def make_hist(image, bins = 2**16):

    dummy = np.copy(image)
    dummy = dummy.flatten()
    dummy = np.sort(dummy)
    width = 1.
    counts, edges, stuff = plt.hist(dummy, bins, histtyp = 'step')

    return counts, edges, width



pixel_data = master[0].data

# pixel_data = pixel_data[0: 1960, 20:1000]
# mask1 = masks.del_grays(pixel_data, 3421)
# mask2 = masks.upper_threshold(pixel_data, 4000)
pixel2 = pixel_data

mask3 = sharp.sharpen(pixel2, 4000, 3466)

pixel2 = np.round(np.multiply(pixel2, mask3))
pixel_ref = pixel2.copy()

plt.imshow(pixel2)
plt.show()
right_hemi, left_hemi, x_perimeter = phot.circle(8)
right_hemi1, left_hemi1, x_perimeter_check = phot.circle(8)

catalog = []

def iter_blob(chart, x_perimeter, iterations):

    xlen = len(chart[0])
    ylen = len(chart)
    radius = len(x_perimeter)

    for run in range(iterations):
        print(run)
        print(np.shape(chart))
        pixel_value = np.max(chart)
        hotspots = np.where(chart == pixel_value)
        for i in range(len(hotspots[0])):
            centre = [hotspots[1][i], hotspots[0][i]]
            neg_exists = phot.neg_areascan(chart, centre, x_perimeter_check)
            if not neg_exists:
                norm_test1, norm_test2 = isradial.test_band(chart, centre, radius, 1)
                if len(norm_test1) > 1 and len(norm_test2) > 1:
                    # print(norm_test1, norm_test2)
                    xfit = np.linspace(0, 1, len(norm_test2))
                    plt.plot(xfit, norm_test2)
                    plt.show()
                    is_normal1 = isradial.normalTest(norm_test1, 0.05)
                    is_normal2 = isradial.normalTest(norm_test2, 0.05)
                    print(isradial.quartile_test(norm_test1, 0.5), 'hi!')
                    print(isradial.quartile_test(norm_test2, 0.5),'hi2!')
                    catalog.append(centre)

            x0 = centre[0] - radius
            x1 = centre[0] + radius

            if x0 < 0.: x0 = 0
            if x1 > xlen: x1 = xlen
            # print(x0, x1, centre[1], hi)
            chart[centre[1]][x0 : x1] = -1.
            for q in range(0, radius, 1):
                y = centre[1] + (q + 1)
                if y >= ylen:
                    y = ylen - 1

                x0 = centre[0] + x_perimeter[q][0]
                x1 = centre[0] + x_perimeter[q][1]
                if x0 < 0.: x0 = 0
                if x1 > xlen: x1 = xlen

                chart[y][x0 : x1] = -1.

                y = centre[1] - (q + 1)
                if y < 0.: y = 0

                chart[y][x0 : x1] = -1.

    return chart

pixel2 = iter_blob(pixel2, x_perimeter, iterations = 500)


#%%%%%%%%%%%%%%%%%%%%%%



centres_mask = np.zeros(np.shape(pixel2))
y_len = len(centres_mask)
x_len = len(centres_mask[0])
print(catalog)
for centre in catalog:

    for q in range(len(right_hemi[0])):
        y = right_hemi[1][q] + centre[1]
        x = right_hemi[0][q] + centre[0]

        if y < y_len and x < x_len:
            centres_mask[y][x] = 2**16 - pixel2[y][x]

    for q in range(len(left_hemi[0])):
        y = left_hemi[1][q] + centre[1]
        x = left_hemi[0][q] + centre[0]

        if y < y_len and x < x_len:
            centres_mask[y][x] = 2**16 - pixel2[y][x]

# negs = np.where(pixel2 < 0.)
# for i in range(len(negs[0])):
#     pixel2[negs[0][i]][negs[1][i]] = 0.
# plt.imshow(pixel2)
# plt.show()

pixel3 = np.add(centres_mask, pixel_ref)
cv2.imwrite('../../test1.png', pixel3)
plt.imshow(pixel3)
plt.show()

# make_hist(pixel_data)
#
# pixel1 = np.multiply(mask1, pixel_data)
#
# histy, edges, width = make_hist(pixel1, 2**16)
# histx = edges + width/2.
# # ### fit skewed gaussian
# domain = [np.where(histx > 3350)[0][0], np.where(histx < 3600)[0][-1]]
# start = domain[0]
# stop = domain[1]
#
# histx_fit = histx[start:stop]
# histy_fit = histy[start:stop]
#
# peak_index = np.where(histy_fit == np.max(histy_fit))[0][0]
# mean_guess = histx_fit[peak_index]
# print(mean_guess)
# sigma_guess = peak_widths(histy_fit, [peak_index])[0][0]
# print(sigma_guess)
# skew_guess = 1.1
# amp_guess = np.max(histy_fit)
# print(amp_guess)
# histx_fit = histx_fit[0:peak_index + 10]
# histy_fit = histy_fit[0:peak_index + 10]
#
#
# # fit_output = fit(skewgauss, histx_fit, histy_fit, initials = [amp_guess, mean_guess, sigma_guess, skew_guess], args = None, xerr = None, yerr = None)
#
# # fit_output = fit(lorentzian, histx_fit, histy_fit, initials = [amp_guess, 3400, sigma_guess], args = None, xerr = None, yerr = None)
# # fit_output = fit(pearson_iv, histx_fit, histy_fit, initials = [amp_guess, 100, 2., 1.5, mean_guess], args = None, xerr = None, yerr = None)
# fit_output = fit(gauss, histx_fit, histy_fit, initials = [amp_guess, mean_guess, sigma_guess/5.4], args = None, xerr = None, yerr = np.sqrt(histy_fit))
#
# #%%%%%%%%%%%%%%
#
#     # amp, alpha, m, v, lam = p
#
#
#
# xfit = np.linspace(histx_fit[0], histx_fit[-1], 1000)
# yfit = gauss(fit_output[0], xfit)
# # yfit = lorentzian([amp_guess, mean_guess, sigma_guess/3], xfit)
#
# # yfit = skewgauss([amp_guess/2., mean_guess, sigma_guess, 10], xfit)
# # yfit = lorentzian(fit_output[0], xfit)
# # yfit = lorentzian([amp_guess, mean_guess, sigma_guess/2.6], xfit)
# # yfit = pearson_iv([amp_guess, 1., 0.5, 0., mean_guess], xfit)
# # yfit = pearson_iv(fit_output[0], xfit)
#
#
# plt.grid(which = 'both')
#
# plt.plot(xfit, yfit)
# # plt.ylim([0, np.max(histy_fit)])
# plt.xlim([3000, 4000])
# plt.xlabel('Pixel Values', fontsize = 14)
# plt.ylabel('Bin Count', fontsize = 14)
# plt.show()
