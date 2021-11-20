#!/usr/bin/env python3
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

pixel_data = master[0].data
mask_ref = sharp.sharpen(pixel_data, 8000, 3450)
pixel_ref = np.round(np.multiply(pixel_data, mask_ref))


# centre_list = np.loadtxt('output/centres_simulation.txt', delimiter = '\t')
centre_list = np.loadtxt('output/centres.txt', delimiter = '\t')

centre_list = centre_list.astype(int)

image = pixel_data
# image = np.load('output/simulation_image.npy')
pixel_ref = image.copy()
# plt.imshow(image)
# plt.show()

fluxes = []
flux_catalog = []
oval_catalog = []
centre_list2 = []

total = len(centre_list)
counter = 0
ellipse_ratios = []

import random

my_list = [i for i in range(0,6149)] # list of integers from 1 to 99
                              # adjust this boundaries to fit your needs
random.shuffle(my_list)
print(my_list)
for index in my_list[0:300]:
    centre = centre_list[index]
    print(counter, 'of', total)
    # totalflux, ovalparam, isvalid, ellipse_ratio = phot.deepscan(image, centre)
    ellipse_ratio = phot.deepscan(image, centre)
    ellipse_ratios.append(ellipse_ratio)
    # if isvalid:
    #     fluxes.append(totalflux)
    #     oval_catalog.append(ovalparam)
    #     centre_list2.append(centre)
    counter += 1
print(ellipse_ratio)
np.savetxt('output/ratios.txt', np.array(ellipse_ratios))



# for index, centre in enumerate(centre_list2):
#
#     image = phot.oval_mask(image, centre, oval_catalog[index], type = 'area', mask_val = 0)
#     pixel_ref = phot.oval_mask(pixel_ref, centre, oval_catalog[index], type = 'edge', mask_val = 2**16)
#
# for index, centre in enumerate(centre_list2):
#
#     ovalparam = oval_catalog[index]
#     image, realnoise = phot.circle_bgnoise(image, centre, round(ovalparam[0]*1.5), ovalparam[4])
#     flux_catalog.append([centre[0], centre[1], fluxes[index] - realnoise])
#
# print(flux_catalog)
# np.savetxt('output/flux_catalog4.txt', np.c_[flux_catalog], delimiter = '\t')
#
# plt.imshow(pixel_ref)
# plt.xlabel('X pixels', fontsize = 14)
# plt.ylabel('Y pixels', fontsize = 14)
#
# plt.show()


#
# for index, centre in enumerate(centre_list):
#     # print(centre)
#     realflux, error = phot.fluxscan(pixel_data, centre, flux_peri, noise_peri)
#     flux_catalog.append(np.append(centre_list[index], realflux))
#
# flux_catalog = np.array(flux_catalog)
# np.savetxt('output/flux_catalog.txt', np.c_[flux_catalog], delimiter = '\t')
#
#
#
# centres_mask = np.zeros(np.shape(pixel_ref))
# y_len = len(centres_mask)
# x_len = len(centres_mask[0])
# # print(catalog)
# for centre in centre_list:
#     for q in range(len(right_hemi[0])):
#         y = right_hemi[1][q] + centre[1]
#         x = right_hemi[0][q] + centre[0]
#
#         if y < y_len and x < x_len:
#             centres_mask[y][x] = 2**16 - pixel_ref[y][x]
#
#     for q in range(len(left_hemi[0])):
#         y = left_hemi[1][q] + centre[1]
#         x = left_hemi[0][q] + centre[0]
#
#         if y < y_len and x < x_len:
#             centres_mask[y][x] = 2**16 - pixel_ref[y][x]
#
# pixel3 = np.add(centres_mask, pixel_ref)
# print('DETECTED OBJECTS:', len(centre_list))
# # print('REJECTED OBJECTS:', rejected)
# cv2.imwrite('../../test1.png', pixel3)
# plt.imshow(pixel3)
# plt.show()
#
# import main02
