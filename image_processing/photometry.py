#!/usr/bin/env python3

import numpy as np
import sys

from numba import njit



def circle(centre, radius):

    first_oct = [[], []]

    initial = [centre[0] + radius, centre[1]]
    next = [0, 0]
    # print(round(centre[1] + radius/np.sqrt(2)))
    while next[1] < centre[1] + radius/np.sqrt(2) - 1:

        next_x2 = initial[0]**2 - 2*initial[1] - 1
        next_y2 = radius**2 - next_x2
        next = [np.sqrt(next_x2), np.sqrt(next_y2)]
        # print(next)
        for i in range(2): first_oct[i].append(round(next[i]))
        initial = next


    first_quad = [first_oct[0] + list(reversed(first_oct[1])), first_oct[1] + list(reversed(first_oct[0]))]
    right_hemi = [list(reversed(first_quad[0])) + first_quad[0], list(reversed(first_quad[1])) + [-1 * i for i in first_quad[1]]]
    left_hemi = [[-1 * i for i in right_hemi[0]], right_hemi[1]]

    return right_hemi, left_hemi


right_hemi, left_hemi = circle([0, 0], 23)
print(right_hemi)
dummy = [[], []]
# for index, y in right_hemi[1]:
#
#     dummy[1].append(y)
#     if right_hemi[]
