import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from numba import njit

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
