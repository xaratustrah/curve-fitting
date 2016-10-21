#!/usr/bin/env python
"""
Rest gas monitor data
Exporter, Fitter, plotter

Usage:
    python *.xml > results.txt

2016

Xaratustrah


thanks to:

http://mesa.ac.nz/2011/10/python-workshop-i-fitting-a-single-symmetric-peak/
http://stackoverflow.com/a/14460456/5177935
http://stackoverflow.com/a/11507723/5177935

"""

import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as et
from io import BytesIO
import sys, os
from scipy.optimize import curve_fit


def read_data(filename):
    with open(filename, 'rb') as f:
        ba = f.read()
    str = '<xml>' + ba.decode("utf-8") + '</xml>'
    str = str.replace('ProfileReadoutTime_yyyyMMdd_hhmmss_zzz', 'ProfileReadoutStartTime')
    xml_tree_root = et.fromstring(str)

    i = 0
    for elem in xml_tree_root.iter(tag='ProfileData'):
        b = np.genfromtxt(BytesIO(elem.text.encode()), delimiter=";", autostrip=True)
        # here comes the actual calculation instead of break, for every column of data
        break
    return b


# fit function
def gauss_function(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def fit_and_plot(filename):
    y = read_data(filename)[:-1]  # there is one undefined point at the end
    x = np.arange(len(y))

    # defining the 'background' part of the spectrum
    ind_bg_low = (x > min(x)) & (x < 450)
    ind_bg_high = (x > 700.0) & (x < max(x))

    x_bg = np.concatenate((x[ind_bg_low], x[ind_bg_high]))
    y_bg = np.concatenate((y[ind_bg_low], y[ind_bg_high]))

    # fitting the background to a line
    m, c = np.polyfit(x_bg, y_bg, 1)

    # removing fitted background
    background = m * x + c
    y_bg_corr = y - background

    # Estimate for mean and sigma
    mean = y_bg_corr.argmax()
    sigma = 100

    # cut the region
    x_cut = x[mean - 300:mean + 300]
    y_cut = y_bg_corr[mean - 300:mean + 300]

    # Try to fit the result
    popt, pcov = curve_fit(gauss_function, x_cut, y_cut, p0=[1, mean, sigma])

    mean = popt[1]
    sigma = popt[2]
    area = sum(y_cut)

    # Plot with original data
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, y, 'k,', label='Data')
    ax.plot(x, y_bg_corr, 'g,', label='Data-BG')
    ax.plot(x_cut, y_cut, 'b', label='Cut')
    ax.plot(x, gauss_function(x, *popt), 'r', label='Fit')
    ax.set_xlabel('mu = {:0.2e}, sig = {:0.2e}, area = {:0.2e}'.format(mean, sigma, area))
    ax.set_title(os.path.basename(filename))

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper right', shadow=False)

    # Set legend fontsize
    for label in legend.get_texts():
        label.set_fontsize('small')

    plt.grid()
    filename_wo_ext = os.path.splitext(filename)[0]
    plt.savefig('{}.pdf'.format(filename_wo_ext))
    print(' '.join(map(str, popt)), area)


def main():
    for file in sys.argv[1:]:
        fit_and_plot(file)


# -------------

if __name__ == "__main__":
    main()
