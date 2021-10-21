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
                    # plt.plot(xfit, norm_test2)
                    # plt.show()
                    is_normal1 = isradial.normalTest(norm_test1, 0.05)
                    is_normal2 = isradial.normalTest(norm_test2, 0.05)
                    isradial1 = isradial.quartile_test(norm_test1, 0.5)
                    isradial2 = isradial.quartile_test(norm_test2, 0.5)

                    if isradial1 or isradial2:
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

np.savetxt('output/centres.txt', np.c_[catalog], delimiter = '\t')

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

pixel3 = np.add(centres_mask, pixel_ref)
cv2.imwrite('../../test1.png', pixel3)
plt.imshow(pixel3)
plt.show()
