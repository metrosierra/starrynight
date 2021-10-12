# !/Users/jiayangzhang/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2


"""
maxpixel_coords
    find the first coordinate where has the max pixel value
arg
    image - 2D array

output
    1d array: [row, column]
"""

def maxpixel_coords(image):

    # max numbers from each row - sublist
    sublist = list( map(max, image) )
    # max numbers from sublist
    max_pixel = max(sublist)
    print("max pixel value", max_pixel)



    row_counter = 0
    for i in range (0, len(sublist) -1):
        if sublist[i] == max_pixel:
            break
        else:
            row_counter += 1
    # print("row sublist：", image[row_counter])


    column_counter = 0
    for i in range(0, len(image[row_counter])-1):
        if image[row_counter][i] == max_pixel:
            break
        else:
            column_counter += 1


    pixel_coords = [row_counter, column_counter]
    return pixel_coords



"""
rect_mask()
    create a mask with a rectangular area
        inside rect area - set to 1
        outside rect - set to 0

arg
    half_width - hald width of rect
    coords - centre of rect
    (height - image height)

out
    mask
"""
def rect_mask(image, half_width, coords):
    mask = np.zeros(np.shape(image), dtype=image.dtype)
    for row in mask:
        centrex = coords[1]
        row[centrex - half_width : centrex + half_width] = 1
    return mask




"""
contours()
    draw contours onto np.zeros mask

    method:
        simple global thresholding
    style of thresholding:
         cv2.THRESH_BINARY

arg:
    image - from which contours are found
    mask - onto which contours are drawn
    thresVal - the threshold value which is used to classify the
                    pixel values
    maxVal - the maxVal which represents the value to be given
                    if pixel value is more than (sometimes less than) the
                    threshold value
out:
    mask
"""
def drawContours(image, thresVal, maxVal):

    mask = np.zeros(np.shape(image), dtype=image.dtype)

    #  find the contour threshold
    retval, threshold = cv2.threshold(image, thresVal, maxVal, cv2.THRESH_BINARY)
    # convert to uint8 type
    threshold = threshold.astype(np.uint8)

    # find contours
    contours, hier = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    cv2.drawContours(mask, contours, -1,(255), 3)

    return mask
