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
import simulations as sim


from skimage.draw import ellipse


#
# img = np.zeros((100, 100), dtype=np.uint8)
# rr, cc = ellipse(0, 0, 6, 4, rotation = np.deg2rad(-80))
#
# coordpairs = np.stack((rr, cc), axis = -1)
#
# x_values = set([pair[0] for pair in coordpairs])
# print(x_values)
# x_groups = [[pair[1] for pair in coordpairs if pair[0] == value] for value in x_values]
#
# x_perimeter = np.array([min(x_groups[0]), max(x_groups[0])])
# for group in x_groups[1:]:
#     new = np.array([min(group), max(group)])
#     x_perimeter = np.append(x_perimeter, new)
#
# x_perimeter = x_perimeter.reshape((len(x_groups), 2))
# print(x_perimeter)
#
# centre = [50, 50]
# ybounding = max(x_values)
# for i in range(ybounding):
# # img[x_perimeter[0], x_perimeter[1]] = 1
#     img[i + centre[1], x_perimeter[i][0] + centre[0]] = 1
#     img[i + centre[1], x_perimeter[i][1] + centre[0]] = 1
#
#     img[-i + centre[1], x_perimeter[-i][0] + centre[0]] = 1
#     img[-i + centre[1], x_perimeter[-i][1] + centre[0]] = 1
# img[ybounding + centre[1], x_perimeter[ybounding][0] + centre[0]] = 1
# img[ybounding + centre[1], x_perimeter[ybounding][1] + centre[0]] = 1
# # img[rr + centre[1], cc + centre[0]] = 1
#
# plt.imshow(img)
# plt.show()



rng = np.random.default_rng()
rints = rng.integers(low=0, high=10, size=3)

xlen = 300  # width of heatmap
ylen = 300  # height of heatmap
amp = 64  # increase scale to make larger gaussians
centres = [(100,100),
           (100,300),
           (300,100)] # center points of the gaussians


centre_offset = np.array([0, 0])

set1 = sim.gauss_spots([rng.integers(low=0, high=300, size = 2) for i in range(10)], ylen, xlen, 20)
print(np.sum(set1))
for i in range(4):

    amp = rng.integers(low=10, high=50, size = 1)
    print(amp)
    dummy = sim.gauss_spots([rng.integers(low=0, high=300, size = 2) for i in range(10)], ylen, xlen, amp)
    set1 = np.add(set1, dummy)


pixel2 = set1



# pixel2 = sim.gauss_spots(centres, ylen, xlen, amp)
pixel_ref= pixel2.copy()
plt.imshow(pixel_ref)
plt.show()


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
        print(pixel_value)
        if pixel_value <= 0.: break

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
plt.imshow(pixel2)
plt.show()
np.save('output/simulation_image.npy', pixel_ref)
np.savetxt('output/centres_simulation.txt', np.c_[catalog], delimiter = '\t')
import pipeline2
#%%%%%%%%%%%%%%%%%%%%%%

# centre_list = np.loadtxt('output/centres_simulation.txt', delimiter = '\t')
# centre_list = centre_list.astype(int)
#
# for index, centre in enumerate(centre_list):
#     radius = 6
#     fluxes = [0.]
#     for i in range(20):
#         right_hemi0, left_hemi0, flux_peri = phot.circle(radius)
#         right_hemi1, left_hemi1, noise_peri = phot.circle(radius * 1.5)
#
#         realflux, error = phot.fluxscan(pixel_ref, centre, flux_peri, noise_peri)
#         fluxes.append(realflux)
#         print('std is', np.std(fluxes))
#         print(np.diff(fluxes))
#         print('percent is', np.diff(fluxes)[-1]/realflux)
#
#         print(realflux, error)
#
#         radius += 1
#
#
#     catalog[index].append(realflux)
# np.savetxt('output/flux_catalog_simulation.txt', np.c_[catalog], delimiter = '\t')

#%%%%%%%%%%%%%%%%%%%%%%
#
#
# centres_mask = np.zeros(np.shape(pixel_ref))
# y_len = len(centres_mask)
# x_len = len(centres_mask[0])
# # print(catalog)
# for centre in catalog:
#     for q in range(len(right_hemi0[0])):
#         y = right_hemi0[1][q] + centre[1]
#         x = right_hemi0[0][q] + centre[0]
#
#         if y < y_len and x < x_len:
#             centres_mask[y][x] = - pixel_ref[y][x] + 0.0049
#
#     for q in range(len(left_hemi0[0])):
#         y = left_hemi0[1][q] + centre[1]
#         x = left_hemi0[0][q] + centre[0]
#
#         if y < y_len and x < x_len:
#             centres_mask[y][x] = - pixel_ref[y][x] + 0.0049
#
#
# pixel3 = np.add(centres_mask, pixel_ref)
# print('DETECTED OBJECTS:', len(catalog))
# print('REJECTED OBJECTS:', rejected)
# # cv2.imwrite('../../test1.png', pixel3)
# plt.imshow(pixel3)
# plt.show()
#
# import main02
