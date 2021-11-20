#!/usr/bin/env python3

import numpy as np
import sys

from numba import njit


### sets gray values from left and right edges to 0 (blackest black)
### outputs a mask
def del_grays(image, grayvalue):

    mask = np.ones(np.shape(image))
    for index, line in enumerate(image):
        if line[0] == grayvalue:
            mask[index][0] = 0
            runner = 1
            while runner < len(line) and line[0+runner] == grayvalue:
                mask[index][0+runner] = 0
                runner += 1

        if line[-1] == grayvalue:
            mask[index][-1] = 0
            runner = 1
            while runner < len(line) and line[-(1+runner)] == grayvalue:
                mask[index][-(1+runner)] = 0
                runner += 1
    return mask

#### set dead zones saturated to negative by convention
@njit
def upper_threshold(image, threshold):

    mask = np.ones(np.shape(image))
    coordinates = np.where(image > threshold)
    for i in range(len(coordinates[0])):

        x = coordinates[0][i]
        y = coordinates[1][i]
        mask[x][y] = -1.

    return mask

### sets gray values from left and right edges to 0 (blackest black)
### outputs a mask
def del_grays(image, grayvalue):

    mask = np.ones(np.shape(image))
    for index, line in enumerate(image):
        if line[0] == grayvalue:
            mask[index][0] = 0
            runner = 1
            while runner < len(line) and line[0+runner] == grayvalue:
                mask[index][0+runner] = 0
                runner += 1

        if line[-1] == grayvalue:
            mask[index][-1] = 0
            runner = 1
            while runner < len(line) and line[-(1+runner)] == grayvalue:
                mask[index][-(1+runner)] = 0
                runner += 1
    return mask

@njit
def sharpen_func(pixel, thresWhite, thresBlack):
    pixel = (pixel - thresBlack) * ((2**16 - 1)/(thresWhite - thresBlack))
    return pixel

@njit
def sharpen(img, thresWhite, thresBlack):
    """
    returns a mask that can sharpen the img

    input
        thresWhite - above which will be set to -1
        thresBlack - above which will be set to 0
    """
    mask = np.zeros(np.shape(img))

    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):

            if img[i][j] <= thresBlack:
                mask[i][j] = 0
            elif img[i][j] >= thresWhite:
                mask[i][j] = -1
            else:
                temp = sharpen_func(img[i][j], thresWhite, thresBlack)
                mask[i][j] = temp / img[i][j]

    return mask
