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

@njit
def upper_threshold(image, threshold):

    mask = np.ones(np.shape(image))
    coordinates = np.where(image > threshold)
    for i in range(len(coordinates[0])):

        x = coordinates[0][i]
        y = coordinates[1][i]
        mask[x][y] = 0

    return mask
