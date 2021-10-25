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
pixel2 = pixel_data[centre_offset[1]: 4422, centre_offset[0]:2364]
# mask1 = masks.del_grays(pixel_data, 3421)
# mask2 = masks.upper_threshold(pixel_data, 4000)
# pixel2 = pixel_data

mask3 = sharp.sharpen(pixel2, 6500, 3420)

pixel2 = np.round(np.multiply(pixel2, mask3))

mask_ref = sharp.sharpen(pixel_data, 6500, 3420)
pixel_ref = np.round(np.multiply(pixel_data, mask_ref))


cv2.imwrite('../../test0.png', pixel2)

# plt.imshow(pixel2)
# plt.show()
right_hemi, left_hemi, x_perimeter = phot.circle(6)
right_hemi1, left_hemi1, x_perimeter_check = phot.circle(6)


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


pixel2, catalog, rejected = iter_blob(pixel2, x_perimeter, iterations = 2000)

np.savetxt('output/centres.txt', np.c_[catalog], delimiter = '\t')

#%%%%%%%%%%%%%%%%%%%%%%
right_hemi, left_hemi, flux_peri = phot.circle(6)
right_hemi, left_hemi, noise_peri = phot.circle(9)

centre_list = np.loadtxt('output/centres.txt', delimiter = '\t')
centre_list = centre_list.astype(int)
fluxes = []
for index, centre in enumerate(centre_list):
    # print(centre)
    realflux, error = phot.fluxscan(pixel_data, centre, flux_peri, noise_peri)

    catalog[index].append(realflux)
np.savetxt('output/flux_catalog.txt', np.c_[catalog], delimiter = '\t')

#%%%%%%%%%%%%%%%%%%%%%%


centres_mask = np.zeros(np.shape(pixel_ref))
y_len = len(centres_mask)
x_len = len(centres_mask[0])
# print(catalog)
for centre in catalog:
    for q in range(len(right_hemi[0])):
        y = right_hemi[1][q] + centre[1]
        x = right_hemi[0][q] + centre[0]

        if y < y_len and x < x_len:
            centres_mask[y][x] = 2**16 - pixel_ref[y][x]

    for q in range(len(left_hemi[0])):
        y = left_hemi[1][q] + centre[1]
        x = left_hemi[0][q] + centre[0]

        if y < y_len and x < x_len:
            centres_mask[y][x] = 2**16 - pixel_ref[y][x]

pixel3 = np.add(centres_mask, pixel_ref)
print('DETECTED OBJECTS:', len(catalog))
print('REJECTED OBJECTS:', rejected)
cv2.imwrite('../../test1.png', pixel3)
plt.imshow(pixel3)
plt.show()

import main02
