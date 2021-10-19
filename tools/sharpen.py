import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def sharpen_func(pixel, thresWhite, thresBlack):
    pixel = (pixel - thresBlack) * ((2**16 - 1)/(thresWhite - thresBlack))
    return pixel

def sharpen(filepath, thresWhite, thresBlack):
    # file object
    hdulist = fits.open(filepath)
    # data
    img = hdulist[0].data

    thresWhite = thresWhite
    thresBlack = thresBlack

    mask = np.zeros(np.shape(img))

    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):

            if img[i][j] <= thresBlack:
                mask[i][j] = 0
            elif img[i][j] >= thresWhite:
                mask[i][j] = 0
            else:
                temp = sharpen_func(img[i][j], thresWhite, thresBlack)
                mask[i][j] = temp / img[i][j]

    return mask
