import numpy as np
def calibration(counts):
    """
    returns calibrated magnitude, which is the flux of the object

    input
        pixel counts of one object

    output
        calibrated magitude, i.e. the flux


    the instumental zero point MAGZPT is:
    MAGZPT = 2.530E+01 / Photometric ZP (mags) for default extinction
    """
#     convert counts into instrumental magnitudes
    magnitude = -2.5 * np.log10(counts)
#     convert instumental magnitudes to calibrated magnitudes
    MAGZPT = 2.530E+01
    magnitude = MAGZPT + magnitude

    return magnitude


def calibration_err(counts, counts_err):
    """
    returns error to calibrated magnitude

    input
        standard deviation of pixel counts of one object

    output
        error to calibrated magnitude
    """
#     error - conversion to instrumental magnitudes
    err = np.absolute( -2.5 * ( counts_err / (counts* np.log(10)))    )
#     error - conversion to calibrated magnitudes
    MAGZRR = 2.000E-02
    err = np.sqrt(  MAGZRR**2 + err**2 )
    return err
