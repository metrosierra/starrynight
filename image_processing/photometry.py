#!/usr/bin/env python3

import numpy as np
import sys

from numba import njit



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

def zero_areascan(image, centre, x_perimeter):

    y_len = len(image)
    x_len = len(image[0])
    toggle = False
    output = []
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
        toggle = 0. in output

        if toggle:
            break

    return toggle


# def value_iterscan(image,)
