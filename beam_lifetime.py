#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Beam lifetime measurement using Schottky

2020

Xaratustrah

"""

from iqtools import *
import numpy as np
from scipy.optimize import curve_fit
import sys
import os


def fit_and_plot(filename):
    filename_base = os.path.basename(filename)
    filename_wo_ext = os.path.splitext(filename)[0]

    nframes = 1000
    lframes = 1024
    sframes = 2200  # starting frame around 23 secs

    iq = TIQData(filename)
    iq.method = 'mtm'
    iq.read(nframes=nframes, lframes=lframes, sframes=sframes)

    xx, yy, zz = iq.get_spectrogram(nframes=nframes, lframes=lframes)

    xa, ya, za = iq.get_averaged_spectrum(xx, yy, zz, 10)
    delta_t = float(ya[2, :1] - ya[1, :1])

    dn = zz[1, :].argmax() - 20
    up = zz[1, :].argmax() + 20
    # print(np.sum(zz[1,dn:up]))
    # plt.plot(zz[1,dn:up])

    peaks = np.array([])
    chpwr = np.array([])

    for i in range(1, int(nframes / 10.0)):
        peaks = np.append(peaks, za[i, :].max())
        chpwr = np.append(chpwr, np.sum(zz[i, dn:up]))

    y = peaks
    #y = chpwr
    x = np.arange(len(y)) * delta_t
    p = [2e-7, 1.2]

    popt, pcov = curve_fit(fit_function, x, y, p0=p)

    # plot with original data
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, y, 'bx', label='Data')
    ax.plot(x, fit_function(x, *popt), 'r', label='Fit')
    ax.set_xlabel('amp = {:0.2e}, tau = {:0.2e}'.format(
        popt[0], popt[1]))
    ax.set_title(filename_base)
    ax.grid()
    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper right', shadow=False)

    # Set legend fontsize
    for label in legend.get_texts():
        label.set_fontsize('small')
    plt.tight_layout()
    plt.savefig('{}.png'.format(filename_wo_ext))
    print(filename_base, 't=', popt[1])  # , 't1/2=', popt[1] * np.log(2))


def fit_function(x, *p):
    """
    Exponential function
    """
    return p[0] * np.exp(-x / p[1])


def main():
    for file in sys.argv[1:]:
        fit_and_plot(file)


# -------------

if __name__ == "__main__":
    main()
