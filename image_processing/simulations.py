#!/usr/bin/env python3

import numpy as np
import scipy.stats as stats

from matplotlib import image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import cv2
from numba import njit


def gauss_spots(centres, ylen, xlen, amp):
    spots = []

    for y, x in centres:
        s = np.identity(2) * amp
        s = np.array([[1., 0.3], [0.2, 1.]]) * amp
        z = stats.multivariate_normal(mean = (x,y), cov=s)
        spots.append(z)

    x = np.arange(0, xlen)
    y = np.arange(0, ylen)
    xx, yy = np.meshgrid(x, y)
    mesh = np.stack([xx.ravel(), yy.ravel()]).T

    zz = sum(point.pdf(mesh) for point in spots)
    chart = zz.reshape([ylen, xlen])
    return chart

W = 800  # width of heatmap
H = 400  # height of heatmap
SCALE = 64  # increase scale to make larger gaussians
CENTERS = [(100,100),
           (100,300),
           (300,100)] # center points of the gaussians

# simulation = gauss_spots(CENTERS, H, W, SCALE)
