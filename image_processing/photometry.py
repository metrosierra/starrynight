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


right_hemi, left_hemi, x_perimeter = circle(23)
