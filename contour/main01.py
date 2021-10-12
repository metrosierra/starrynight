# !/Users/jiayangzhang/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
import contour


# .fits file object
hdulist = fits.open("/Users/jiayangzhang/Documents/Imperial/labs/year3/astrolab/sources/A1_mosaic.fits")

# data
img = hdulist[0].data

# =============================================

# coordinates of max pixel value
coords = contour.maxpixel_coords(img)
print("max pixel coordinates", coords)
# !!! something wrong with the coords function!!!!!!!!!!!!!!!!!!!


# mask - with a rect area
# old max coords = [4,1330]
# new max coords = [657, 2530]
mask = contour.rect_mask(img, 360, [4,1330])

# dot with original image
#     inside rect - orignal value
#     outside rect - 0
img_masked = np.multiply(img, mask)

# new mask - draw contours
mask2 = contour.drawContours(img_masked, 10000, 40000)

# save file
# cv2.imwrite("/Users/jiayangzhang/Documents/Imperial/labs/year3/astro/out/contouredMask.jpg", mask2)
# print('Successfully saved')
# plt.imshow(mask2)
