#!/usr/bin/env python3

import numpy as np
import sys

from scipy.signal import peak_widths
from scipy.special import gamma
from scipy.special import erf
from scipy.odr import *
from scipy import stats
import numpy as np

def gauss(p, x):
    amp, mean, sigma = p
    spread = np.exp((-(x - mean) ** 2.0) / (2 * sigma ** 2.0))
    #skewness = (1 + erf((skew * (x - mean)) / (sigma * np.sqrt(2))))
    return amp * spread

def skewgauss(p, x):
    amp, mean, sigma, skew = p
    spread = np.exp((-(x - mean) ** 2.0) / (2 * sigma ** 2.0))
    skewness = (1 + erf((skew * (x - mean)) / (sigma * np.sqrt(2))))
    return amp * spread * skewness

#unnormalised
def lorentzian(p, x):
    amp, mean, gamma = p
    return amp * (gamma**2) / ((x - mean)**2 + gamma**2)

#unnormalised
def pearson_iv(p, x):
    amp, alpha, m, v, lam = p
    lam2 = lam + (alpha*v)/(2*(m - 1))
    return amp / alpha * ((1 + (x - lam2)/alpha)**(2.))**(-m) * np.exp(-v * np.arctan((x - lam2)/alpha))


def fit(function, x, y, initials, args = None, xerr = 0., yerr = 0.):

    # Create a scipy Model object
    model = Model(function, extra_args = args)
    # Create a RealData object using our initiated data from above. basically declaring all our variables using the RealData command which the scipy package wants
    input = RealData(x, y, sx = xerr, sy = yerr)
    # Set up ODR with the model and data. ODR is orthogonal distance regression (need to google!)
    odr = ODR(input, model, beta0 = initials)

    print('\nRunning fit!')
    # Run the regression.
    output = odr.run()
    print('\nFit done!')
    # output.beta contains the fitted parameters (it's a list, so you can sub it back into function as p!)
    print('\nFitted parameters = ', output.beta)
    print('\nError of parameters =', output.sd_beta)

    if xerr is not 0. or yerr is not 0.:
        #now we can calculate chi-square (if you included errors for fitting, if not it's meaningless)
        chisquare = np.sum((y - function(output.beta, x, args))**2/(yerr**2 + xerr**2))
        chi_reduced = chisquare / (len(x) - len(initials))
        print('\nReduced Chisquare = ', chi_reduced, 'with ',  len(x) - len(initials), 'Degrees of Freedom')

    else:
        chi_reduced = 0.


    return output.beta, output.sd_beta, chi_reduced



### discrete pixel values, 16 bit
def make_hist(image, bins = 2**16):

    dummy = np.copy(image)
    dummy = dummy.flatten()
    dummy = np.sort(dummy)
    width = 1.
    counts, edges, stuff = plt.hist(dummy, bins, histtyp = 'step')

    return counts, edges, width
