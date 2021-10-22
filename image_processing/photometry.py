#!/usr/bin/env python3

import numpy as np
import sys

from numba import njit
import matplotlib.pyplot as plt
import isradial

def circle(radius):

    first_oct = [[], []]

    initial = [radius, 0]
    next = [0, 0]
    # print(round(centre[1] + radius/np.sqrt(2)))
    while next[1] < radius/np.sqrt(2) - 1:

        next_x2 = initial[0]**2 - 2*initial[1] - 1
        next_y2 = radius**2 - next_x2
        next = [np.sqrt(next_x2), np.sqrt(next_y2)]
        # print(next)
        for i in range(2): first_oct[i].append(round(next[i]))
        initial = next


    first_quad = [first_oct[0] + list(reversed(first_oct[1])), first_oct[1] + list(reversed(first_oct[0]))]
    right_hemi = [list(reversed(first_quad[0])) + first_quad[0], list(reversed(first_quad[1])) + [-1 * i for i in first_quad[1]]]
    left_hemi = [[-1 * i for i in right_hemi[0]], right_hemi[1]]

    y_hemi = [[right_hemi[0][i], right_hemi[1][i]] for i in range(len(right_hemi[0]))]
    x_values = set([list[0] for list in y_hemi])
    x_groups = [[list[1] for list in y_hemi if list[0] == value] for value in x_values]
    x_perimeter = [[min(group), max(group)] for group in x_groups]

    return right_hemi, left_hemi, x_perimeter

#takes centre as (x, y)
def linescan(image, centre, sigma):

    x0 = centre[0] - sigma
    x1 = centre[0] + sigma
    xline_values = image[centre[1], x0 : x1]
    xline = np.array([i for i in range(x0, x1, 1)])


    y0 = centre[1] - sigma
    y1 = centre[1] + sigma
    yline_values = image[y0 : y1, centre[0]]
    yline = np.array([i for i in range(y0, y1, 1)])

    return xline, xline_values, yline, yline_values

def areascan(image, centre, x_perimeter):

    output = []
    radius = len(x_perimeter)
    x0 = centre[0] - radius
    x1 = centre[0] + radius
    output += list(image[centre[1], x0 : x1])
    for q in range(0, radius):
        y = centre[1] + (q + 1)
        x0 = centre[0] + x_perimeter[q][0]
        x1 = centre[0] + x_perimeter[q][1]
        output += list(image[y, x0 : x1])

        y = centre[1] - (q + 1)
        output += list(image[y, x0 : x1])

    return output

def ischosen(image, centre, noise_radius, x_perimeter_check):

    ischosen = False

    neg_exists = neg_areascan(image, centre, x_perimeter_check)
    if neg_exists:
        return ischosen

    else:

        norm_test1, norm_test2 = isradial.test_band(image, centre, noise_radius, 1)
        if len(norm_test1) > 1 and len(norm_test2) > 1:
            isradial_threshold = 0.5
            isradial1 = isradial.quartile_test(norm_test1, isradial_threshold)
            isradial2 = isradial.quartile_test(norm_test2, isradial_threshold)

            if isradial1 or isradial2:
                ischosen = True

        return ischosen


### noise compensated as well
def fluxscan(image, centre, sig_perimeter, noise_perimeter):

    xlen = len(image[0])
    ylen = len(image)

    sig_radius = sig_perimeter[0][1]
    noise_radius = noise_perimeter[0][1]

    x0 = centre[0] - sig_radius
    x1 = centre[0] + sig_radius

    rawflux = np.array([])
    rawflux  = np.append(rawflux, image[centre[1], x0:x1])

    q0 = centre[0] - noise_radius
    q1 = centre[0] + noise_radius
    if q1 >= ylen: q0 = ylen - 1

    rawnoise = np.array([])
    rawnoise = np.append(rawnoise, image[centre[1], q0:x0])
    rawnoise = np.append(rawnoise, image[centre[1], x1:q1])
    for y in range(1, noise_radius):
        y0 = centre[1] + y
        y1 = centre[1] - y

        q0 = noise_perimeter[y][0] + centre[0]
        q1 = noise_perimeter[y][1] + centre[0]

        if y < sig_radius:
            x0 = sig_perimeter[y][0] + centre[0]
            x1 = sig_perimeter[y][1] + centre[0]
            print(x0, x1)
            for ytemp in [y0, y1]:
                rawflux  = np.append(rawflux, image[ytemp, x0:x1])
                rawnoise = np.append(rawnoise, image[ytemp, q0:x0])
                rawnoise = np.append(rawnoise, image[ytemp, x1:q1])

        elif y >= sig_radius:
            for ytemp in [y0, y1]:
                rawnoise = np.append(rawnoise, image[ytemp, q0:q1])

    totalflux = np.sum(rawflux)
    scaled_noise = np.sum(rawnoise)/len(rawnoise) * len(rawflux)
    realflux = totalflux - scaled_noise
    error = np.std(rawnoise)/np.sum(rawnoise) * scaled_noise

    # counts, edges, stuff = plt.hist(rawnoise, bins = len(rawnoise))
    # plt.show()
    return realflux, error



def neg_areascan(image, centre, x_perimeter):

    y_len = len(image)
    x_len = len(image[0])
    toggle = False
    output = [0.]
    radius = len(x_perimeter)
    x0 = centre[0] - radius
    x1 = centre[0] + radius
    output += list(image[centre[1], x0 : x1])
    for q in range(0, radius):
        y = centre[1] + q
        if y < y_len:
            x0 = centre[0] + x_perimeter[q][0]
            x1 = centre[0] + x_perimeter[q][1]
            output += list(image[y, x0 : x1])

        y = centre[1] - q
        output += list(image[y, x0 : x1])
        toggle = np.any(np.array(output) < 0.)

        if toggle:
            break

    return toggle


# def value_iterscan(image,)
