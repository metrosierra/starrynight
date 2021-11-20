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
#%%%%%%%%%%%%%%%%%%%%%%

centre_offset = np.array([218, 222])
# centre_offset = np.array([570, 3560])

pixel2 = pixel_data[centre_offset[1]: 4422, centre_offset[0] : 2364]
# mask1 = masks.del_grays(pixel_data, 3421)
# mask2 = masks.upper_threshold(pixel_data, 4000)
# pixel2 = pixel_data

mask3 = sharp.sharpen(pixel2, 8000, 3450)

pixel2 = np.round(np.multiply(pixel2, mask3))

mask_ref = sharp.sharpen(pixel_data, 8000, 3450)
pixel_ref = np.round(np.multiply(pixel_data, mask_ref))


cv2.imwrite('../../test0.png', pixel2)

# plt.imshow(pixel2)
# plt.show()
right_hemi, left_hemi, x_perimeter = phot.circle(10)
right_hemi1, left_hemi1, x_perimeter_check = phot.circle(10)


@njit
def iter_blob(chart, x_perimeter, iterations):

    catalog = []
    rejected = 0
    xlen = len(chart[0])
    ylen = len(chart)
    radius = len(x_perimeter)

    for run in range(iterations):
        print('Cycle', run)
        pixel_value = np.max(chart)

        if pixel_value == 0.: break
        hotspots = np.where(chart == pixel_value)
        for i in range(len(hotspots[0])):
            centre = [hotspots[1][i], hotspots[0][i]]

            if phot.ischosen(chart, centre, round(radius * 1.5), x_perimeter_check):
                catalog.append([centre[0] + centre_offset[0], centre[1] + centre_offset[1]])

            else:
                rejected += 1

            x0 = centre[0] - radius
            x1 = centre[0] + radius

            if x0 < 0.: x0 = 0
            if x1 > xlen: x1 = xlen
            # print(x0, x1, centre[1], hi)
            chart[centre[1]][x0 : x1] = -1.
            for q in range(0, radius, 1):
                y = centre[1] + (q + 1)
                if y >= ylen: y = ylen - 1

                x0 = centre[0] + x_perimeter[q][0]
                x1 = centre[0] + x_perimeter[q][1]
                if x0 < 0.: x0 = 0
                if x1 > xlen: x1 = xlen

                chart[y][x0 : x1] = -1.

                y = centre[1] - (q + 1)
                if y < 0.: y = 0

                chart[y][x0 : x1] = -1.

    return chart, catalog, rejected


pixel2, catalog, rejected = iter_blob(pixel2, x_perimeter, iterations = 8000)
print(len(catalog), 'OBJECTS DETECTED')
np.savetxt('output/centres.txt', np.c_[catalog], delimiter = '\t')


yyxx, boundrad, x_perimeter = phot.oval(7, 7, 0)

for index, centre in enumerate(catalog):

    pixel_ref = phot.oval_mask(pixel_ref, centre, [5, 5, boundrad, x_perimeter, 10], type = 'edge', mask_val = round(np.max(pixel_ref)/2))

plt.imshow(pixel_ref)
plt.show()
