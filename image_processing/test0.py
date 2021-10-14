#!/usr/bin/env python3

from astropy.io import fits
import numpy as np
import sys
from matplotlib import image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from scipy.signal import peak_widths
from scipy.special import gamma
from scipy.special import erf
from scipy.odr import *
from scipy import stats


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


def fit(function, x, y, initials, args = None, xerr = None, yerr = None):

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

    if xerr is not None and yerr is not None:
        #now we can calculate chi-square (if you included errors for fitting, if not it's meaningless)
        chisquare = np.sum((y - function(output.beta, x, args))**2/yerr**2)
        chi_reduced = chisquare / (len(x) - len(initials))
        print('\nReduced Chisquare = ', chi_reduced, 'with ',  len(x) - len(initials), 'Degrees of Freedom')

    else:
        chi_reduced = 0.


    return output.beta, output.sd_beta, chi_reduced





master = fits.open('../A1_mosaic.fits')
metadata = master[0].header
# print(metadata)
metatxt = open('metadata.txt', 'w')
metatxt.write(str(metadata))


pixel_data = master[0].data

### sets gray values from left and right edges to 0 (blackest black)
### outputs a mask
def del_grays(image, grayvalue):

    mask = np.ones(np.shape(image))
    for index, line in enumerate(image):
        if line[0] == grayvalue:
            mask[index][0] = 0
            runner = 1
            while runner < len(line) and line[0+runner] == grayvalue:
                mask[index][0+runner] = 0
                runner += 1

        if line[-1] == grayvalue:
            mask[index][-1] = 0
            runner = 1
            while runner < len(line) and line[-(1+runner)] == grayvalue:
                mask[index][-(1+runner)] = 0
                runner += 1
    return mask


### discrete pixel values, 16 bit
def make_hist(image, bins = 2**16):

    dummy = np.copy(image)
    dummy = dummy.flatten()
    dummy = np.sort(dummy)
    width = 1.
    counts, edges, stuff = plt.hist(dummy, bins, histtype='step')

    return counts, edges, width


mask1 = del_grays(pixel_data, 3421)
# make_hist(pixel_data)

pixel1 = np.multiply(mask1, pixel_data)

histy, edges, width = make_hist(pixel1, 2**16)
histx = edges + width/2.
# ### fit skewed gaussian
domain = [np.where(histx > 3350)[0][0], np.where(histx < 3600)[0][-1]]
start = domain[0]
stop = domain[1]

histx_fit = histx[start:stop]
histy_fit = histy[start:stop]

peak_index = np.where(histy_fit == np.max(histy_fit))[0][0]
mean_guess = histx_fit[peak_index]
print(mean_guess)
sigma_guess = peak_widths(histy_fit, [peak_index])[0][0]
print(sigma_guess)
skew_guess = 1.1
amp_guess = np.max(histy_fit)
print(amp_guess)
histx_fit = histx_fit[0:peak_index + 10]
histy_fit = histy_fit[0:peak_index + 10]


# fit_output = fit(skewgauss, histx_fit, histy_fit, initials = [amp_guess, mean_guess, sigma_guess, skew_guess], args = None, xerr = None, yerr = None)

# fit_output = fit(lorentzian, histx_fit, histy_fit, initials = [amp_guess, 3400, sigma_guess], args = None, xerr = None, yerr = None)
# fit_output = fit(pearson_iv, histx_fit, histy_fit, initials = [amp_guess, 100, 2., 1.5, mean_guess], args = None, xerr = None, yerr = None)
fit_output = fit(gauss, histx_fit, histy_fit, initials = [amp_guess, mean_guess, sigma_guess/5.4], args = None, xerr = None, yerr = np.sqrt(histy_fit))

#%%%%%%%%%%%%%%

    # amp, alpha, m, v, lam = p



xfit = np.linspace(histx_fit[0], histx_fit[-1], 1000)
yfit = gauss(fit_output[0], xfit)
# yfit = lorentzian([amp_guess, mean_guess, sigma_guess/3], xfit)

# yfit = skewgauss([amp_guess/2., mean_guess, sigma_guess, 10], xfit)
# yfit = lorentzian(fit_output[0], xfit)
# yfit = lorentzian([amp_guess, mean_guess, sigma_guess/2.6], xfit)
# yfit = pearson_iv([amp_guess, 1., 0.5, 0., mean_guess], xfit)
# yfit = pearson_iv(fit_output[0], xfit)


plt.grid(which = 'both')

plt.plot(xfit, yfit)
# plt.ylim([0, np.max(histy_fit)])
plt.xlim([3000, 4000])
plt.xlabel('Pixel Values', fontsize = 14)
plt.ylabel('Bin Count', fontsize = 14)
plt.show()
