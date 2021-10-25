#!/usr/bin/env python3

import numpy as np
import calibration
import plotter

data = np.loadtxt("output/flux_catalog.txt", skiprows=1, delimiter='\t')

flux_list = []
for index, row in enumerate(data):
    flux = row[2]
    if flux > 0.: flux_list.append(flux)

mag_list = []
for flux in flux_list:
    mag = calibration.calibration(flux)
    mag_list.append(mag)

plotter.mag_plot(mag_list,200)
