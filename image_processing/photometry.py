#!/usr/bin/env python3

import numpy as np
import sys
from skimage.draw import ellipse

from numba import njit
import matplotlib.pyplot as plt

import stat_methods as stat
import isradial

from numba import njit

def oval(yrad, xrad, degrees):
    rr, cc = ellipse(0, 0, yrad, xrad, rotation = np.deg2rad(degrees))
    coordpairs = np.stack((rr, cc), axis = -1)
    rrcc = np.stack((rr, cc))

    x_values = set([pair[0] for pair in coordpairs])
    x_groups = [[pair[1] for pair in coordpairs if pair[0] == value] for value in x_values]

    x_perimeter = np.array([min(x_groups[0]), max(x_groups[0])])
    for group in x_groups[1:]:
        new = np.array([min(group), max(group)])
        x_perimeter = np.append(x_perimeter, new)

    x_perimeter = x_perimeter.reshape((len(x_groups), 2))
    boundrad = max(x_values)

    return rrcc, boundrad, x_perimeter


def oval_mask(image, centre, ovalparam, type = 'area', mask_val = -1):

    majorlen, minorlen, boundrad, x_perimeter, size = ovalparam
    for i in range(-boundrad + 1, boundrad + 1, 1):
    # img[x_perimeter[0], x_perimeter[1]] = 1

        x0 = x_perimeter[i][0] + centre[0]
        x1 = x_perimeter[i][1] + centre[0]
        y = i + centre[1]

        if type == 'area':
            image[y][x0 : x1] = mask_val

        elif type == 'edge':
            image[y][x0] = mask_val
            image[y][x1] = mask_val

    return image

##only apply on zero treated image!!!
def circle_bgnoise(image, centre, radius, size):

    ylen = len(image)
    xlen = len(image[0])

    yyxx, boundrad, perimeter = oval(radius, radius, 0)
    # image = oval_mask(image, centre, [radius, radius, boundrad, perimeter, 2], type = 'edge', mask_val = 0)

    rawnoise = np.array([])
    for i in range(-boundrad + 1, boundrad + 1, 1):

        x0 = perimeter[i][0] + centre[0]
        x1 = perimeter[i][1] + centre[0]

        if x0 > xlen: x0 = xlen - 1
        elif x0 < xlen: x0 = 0
        if x1 > xlen: x1 = xlen - 1
        elif x1 < xlen: x1 = 0

        y = i + centre[1]
        if y > ylen: y = ylen - 1
        if y < ylen: y = 0

        rawnoise = np.append(rawnoise, image[y, x0: x1])

    # rawnoise = image[yyxx[0] + centre[1], yyxx[1] + centre[0]]
    rawnoise = np.delete(rawnoise, np.where(rawnoise == 0))

    realnoise = size * np.sum(rawnoise)/len(rawnoise)

    return image, realnoise



@njit
def circle(radius):

    tempx = []
    tempy = []

    initial = [float(radius), 0.]
    next = [0., 0.]
    # print(round(centre[1] + radius/np.sqrt(2)))
    while next[1] < radius/np.sqrt(2) - 1:

        next_x2 = initial[0]**2 - 2*initial[1] - 1
        next_y2 = radius**2 - next_x2
        next = [np.sqrt(next_x2), np.sqrt(next_y2)]

        tempx.append(round(np.sqrt(next_x2)))
        tempy.append(round(np.sqrt(next_y2)))

        initial = next

    tempx = np.array(tempx)
    tempy = np.array(tempy)

    firstquad_x = np.append(tempx, np.flip(tempy))
    firstquad_y = np.append(tempy, np.flip(tempx))

    righthemi_x = np.append(np.flip(firstquad_x), firstquad_x)
    righthemi_y = np.append(np.flip(firstquad_y), firstquad_y * -1)
    right_hemi = np.stack((righthemi_x, righthemi_y))

    lefthemi_x = righthemi_x * -1
    lefthemi_y = righthemi_y
    left_hemi = np.stack((lefthemi_x, lefthemi_y))

    coordpairs = np.stack((righthemi_x, righthemi_y), axis = -1)

    x_values = set([pair[0] for pair in coordpairs])
    x_groups = [[pair[1] for pair in coordpairs if pair[0] == value] for value in x_values]

    x_perimeter = np.array([min(x_groups[0]), max(x_groups[0])])
    for group in x_groups[1:]:
        new = np.array([min(group), max(group)])
        x_perimeter = np.append(x_perimeter, new)

    x_perimeter = x_perimeter.reshape((len(x_groups), 2))

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

@njit
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

def deepscan(image, centre):

    maxflux = 0
    maxdegree = -90
    for degree in range(90, -90, -1):
        chart = image.copy()
        yyxx, boundrad, x_perimeter = oval(7, 3, degree)

        rawflux = np.array([0])
        for i in range(-boundrad + 1, boundrad + 1, 1):
        # img[x_perimeter[0], x_perimeter[1]] = 1

            x0 = x_perimeter[i][0] + centre[0]
            x1 = x_perimeter[i][1] + centre[0]
            rawflux = np.append(rawflux, image[i + centre[1], x0: x1])

        nowflux = np.sum(rawflux)
        if nowflux > maxflux:
            maxflux = nowflux
            maxdegree = degree

    print('slant degree', maxdegree)
    majorband = isradial.slant_radial(image, centre, 10, maxdegree)
    offset_guess = np.min(majorband[1])
    mean_guess = majorband[0][0]
    amp_guess = np.max(majorband[1])
    var_guess = (np.dot(majorband[0]**2, majorband[1]-offset_guess) - mean_guess**2)/np.max(majorband[1])**2
    print(np.sqrt(var_guess),'std guess')

    initials = [amp_guess, mean_guess+0.000001, var_guess, offset_guess]
    output1 = stat.fit(stat.gauss, majorband[0].astype(float), majorband[1], initials, yerr = 0.05*(majorband[1]))
    # print(output1)
    majorlen = round(4 * np.sqrt(abs(output1[0][2])))

    minorband = isradial.slant_radial(image, centre, 8, maxdegree - 90.)
    offset_guess = np.min(minorband[1])
    mean_guess = minorband[0][0]
    var_guess = (np.dot(minorband[0]**2, minorband[1]-offset_guess) - mean_guess**2)/np.max(minorband[1])**2
    print(np.sqrt(var_guess),'std guess')

    initials = [amp_guess, mean_guess+0.000001, var_guess, offset_guess]
    output2 = stat.fit(stat.gauss, minorband[0].astype(float), minorband[1], initials, yerr = 0.05*(minorband[1]))

    minorlen = round(4 * np.sqrt(abs(output2[0][2])))

    if majorlen == 0. and not minorlen == 0.:
        majorlen = minorlen

    if minorlen == 0. and not majorlen == 0.:
        minorlen = majorlen

    if majorlen <= 2:
        majorlen = 6

    if minorlen <= 2:
        minorlen = 6

    #to ensure tip is pointed the right way
    ### probably runaway fit
    if majorlen > 20:
        majorlen = 8

    if minorlen > 20:
        minorlen = 8

    if majorlen < minorlen:
        majorlen, minorlen = minorlen, majorlen

    ellipse_ratio = majorlen/minorlen

    print(majorlen, minorlen, 'majorminor')
    print(ellipse_ratio, 'ratio')
    # xfit = np.linspace(np.min(minorband[0]), np.max(minorband[0]), 1000)
    # yfit = stat.gauss(output2[0], xfit)
    # plt.plot(xfit, yfit)
    # plt.plot(minorband[0], minorband[1])
    # plt.show()
    # xfit = np.linspace(np.min(majorband[0]), np.max(majorband[0]), 1000)
    # yfit = stat.gauss(output1[0], xfit)
    # plt.plot(xfit, yfit)
    # plt.plot(majorband[0], majorband[1])
    # plt.show()
    #
    # yyxx, boundrad, x_perimeter = oval(majorlen, minorlen, maxdegree)
    #
    # ymin = centre[1] - boundrad
    # ymax = centre[1] + boundrad
    # xmin = centre[0] - round(majorlen*1.2)
    # xmax = centre[0] + round(majorlen*1.2)
    #
    # if ymin < 0 or ymax > len(image) or xmin < 0 or xmax > len(image[0]):
    #
    #     totalflux = 0.
    #     ovalparam = []
    #     isvalid = False
    #     print('FALSE!!!!', centre)
    #
    # else:
    #     isvalid = True
    #     realflux = np.array([])
    #     for i in range(-boundrad + 1, boundrad + 1, 1):
    #     # img[x_perimeter[0], x_perimeter[1]] = 1
    #
    #         x0 = x_perimeter[i][0] + centre[0]
    #         x1 = x_perimeter[i][1] + centre[0]
    #         y = i + centre[1]
    #         realflux = np.append(realflux, image[y, x0: x1])
    #         image[y][x0] = 0.01
    #         image[y][x1] = 0.01
    #
    #     totalflux = np.sum(realflux)
    #     ovalparam = [majorlen, minorlen, boundrad, x_perimeter, len(realflux)]
    #
    #     print(totalflux, 'TOTALFLUX')
    #     print(isvalid)
    return ellipse_ratio#, totalflux, ovalparam, isvalid


### noise compensated as well
@njit
def fluxscan(image, centre, sig_perimeter, noise_perimeter):

    xlen = len(image[0])
    ylen = len(image)

    sig_radius = sig_perimeter[0][1]
    noise_radius = noise_perimeter[0][1]

    x0 = centre[0] - sig_radius
    x1 = centre[0] + sig_radius

    rawflux  = image[centre[1], x0:x1]

    q0 = centre[0] - noise_radius
    q1 = centre[0] + noise_radius
    if q1 >= ylen: q0 = ylen - 1

    rawnoise = image[centre[1], q0:x0]
    rawnoise = np.append(rawnoise, image[centre[1], x1:q1])
    for y in range(1, noise_radius):
        y0 = centre[1] + y
        y1 = centre[1] - y

        q0 = noise_perimeter[y][0] + centre[0]
        q1 = noise_perimeter[y][1] + centre[0]

        if y < sig_radius:
            x0 = sig_perimeter[y][0] + centre[0]
            x1 = sig_perimeter[y][1] + centre[0]
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


@njit
def neg_areascan(image, centre, x_perimeter):

    y_len = len(image)
    x_len = len(image[0])
    toggle = False
    radius = len(x_perimeter)
    x0 = centre[0] - int(radius)
    x1 = centre[0] + int(radius)

    output = image[centre[1], x0 : x1]
    for q in range(0, radius):
        y = centre[1] + q
        if y < y_len:
            x0 = centre[0] + x_perimeter[q][0]
            x1 = centre[0] + x_perimeter[q][1]
            output = np.append(output, image[y, x0 : x1])

        y = centre[1] - q
        output = np.append(output, image[y, x0 : x1])
        toggle = np.any(output < 0.)

        if toggle:
            break

    return toggle


@njit
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

# right_hemi, left_hemi, x_perimeter = circle(8)
# print(x_perimeter)
# def value_iterscan(image,)
