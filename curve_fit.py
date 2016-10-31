#!/usr/bin/env python
"""
Rest gas monitor data
Exporter, Fitter, plotter

Usage:
    python *.xml > results.txt

2016

Xaratustrah


Fitting hints thanks to:

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

    data_array = np.array([])
    for elem in xml_tree_root.iter(tag='ProfileData'):
        b = np.genfromtxt(BytesIO(elem.text.encode()), delimiter=";", autostrip=True)
        data_array = np.append(data_array, b[:-1])  # there is one undefined point at the end

    data_matrix = np.reshape(data_array, (int(np.shape(data_array)[0] / 1280), 1280))
    data_matrix_mean = np.mean(data_matrix, axis=0)

    return data_matrix_mean, data_matrix, data_array


def fit_function(x, *p):
    """
    Line + Gaussian
    """
    A0, A1, A2, B2, C2 = p
    return A0 + A1 * x + A2 * np.exp(-(x - B2) ** 2 / (2. * C2 ** 2))


def fit_and_plot(filename, range, sigma_estimate):
    filename_base = os.path.basename(filename)
    filename_wo_ext = os.path.splitext(filename)[0]

    y, _, _ = read_data(filename)
    x = np.arange(len(y))

    # Estimate for mean and sigma
    mean = y.argmax()
    sigma = 100
    offset = y[mean - 250]
    slope = 1
    amp = 1

    # defining the fitting region
    data_cut = (x > mean - 250) & (x < mean + 250)

    # fit
    popt, pcov = curve_fit(fit_function, x[data_cut], y[data_cut], p0=[offset, slope, amp, mean, sigma])

    # Get the area
    area = sum(fit_function(x, *popt))

    # plot with original data
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, y, 'k,', label='Data')
    ax.plot(x[data_cut], fit_function(x[data_cut], *popt), 'r', label='Fit')
    ax.set_xlabel('mu = {:0.2e}, sig = {:0.2e}, area = {:0.2e}'.format(mean, sigma_estimate, area))
    ax.set_title(filename_base)

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper right', shadow=False)

    # Set legend fontsize
    for label in legend.get_texts():
        label.set_fontsize('small')

    plt.grid()
    plt.savefig('{}.pdf'.format(filename_wo_ext))
    print(filename_base, ' '.join(map(str, popt)), area)


## ====================================
# old fit procedures, not used
## ====================================

def gauss_function(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def fit_and_plot2(filename, range, sigma_estimate):
    filename_base = os.path.basename(filename)
    filename_wo_ext = os.path.splitext(filename)[0]

    y, _, _ = read_data(filename)
    x = np.arange(len(y))

    # Estimate for mean and sigma
    mean = y.argmax()

    # defining the 'background' part of the spectrum
    ind_bg_low = (x > min(x)) & (x < mean - range)
    ind_bg_high = (x > mean + range) & (x < max(x))

    x_bg = np.concatenate((x[ind_bg_low], x[ind_bg_high]))
    y_bg = np.concatenate((y[ind_bg_low], y[ind_bg_high]))

    # fitting the background to a line
    m, c = np.polyfit(x_bg, y_bg, 1)

    # subtract fitted background
    background = m * x + c
    y_bg_corr = y - background

    # cut the region
    x_cut = x[mean - range:mean + range]
    y_cut = y_bg_corr[mean - range:mean + range]

    # Try to fit the result
    popt, pcov = curve_fit(gauss_function, x_cut, y_cut, p0=[1, mean, sigma_estimate])

    # Get the area
    area = sum(gauss_function(x, *popt))

    # Plot with original data
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, y, 'k,', label='Data')
    ax.plot(x, y_bg_corr, 'g,', label='Data-BG')
    ax.plot(x_cut, y_cut, 'b', label='Cut')
    ax.plot(x, gauss_function(x, *popt), 'r', label='Fit')
    ax.set_xlabel('mu = {:0.2e}, sig = {:0.2e}, area = {:0.2e}'.format(mean, sigma_estimate, area))
    ax.set_title(filename_base)

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper right', shadow=False)

    # Set legend fontsize
    for label in legend.get_texts():
        label.set_fontsize('small')

    plt.grid()
    plt.savefig('{}.pdf'.format(filename_wo_ext))
    print(filename_base, ' '.join(map(str, popt)), area)


def main():
    for file in sys.argv[1:]:
        fit_and_plot(file, 200, 100)


# -------------

if __name__ == "__main__":
    main()
