#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Curve fit multiple Schottky data

2017

Xaratustrah

"""

from iqtools import *
from scipy.optimize import curve_fit


def fit_function(x, *p):
    """
    Gaussian
    """
    return p[0] + p[1] * np.exp(-(x - p[2]) ** 2 / (2. * p[3] ** 2))


def get_spectrum(iq_data):
    # whatever procedure needed can be put here
    f, p, _ = iq_data.get_fft()
    return f, p


def main():
    for filename in sys.argv[1:]:

        # Process filenames
        print('Processing file: ' + filename)
        filename_base = os.path.basename(filename)
        filename_wo_ext = os.path.splitext(filename)[0]

        # Create Instance
        d = TIQData(filename)
        d.read_samples(100 * 1024, 1000)
        f, p = get_spectrum(d)
        mean, sigma, mhm, phm = d.get_sigma_estimate(f, p)

        # set initial params
        offset = mean
        amp = p[mean]
        params = [offset, amp, mean, sigma]

        # define dummy axis
        x = np.arange(len(p))
        # define cut
        data_cut = (x > mean - 4 * sigma) & (x < mean + 4 * sigma)

        # Do Fit
        popt, pcov = curve_fit(fit_function, x[data_cut], p[data_cut], p0=params)

        # Get the area
        area = sum(fit_function(x, *popt))

        # plot with original data
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(f, p, 'b,', label='Data')
        ax.plot(f[data_cut], fit_function(x[data_cut], *popt), 'r', label='Fit')
        ax.set_xlabel('mu = {:0.2e}, sig = {:0.2e}, area = {:0.2e}'.format(mean, sigma, area))
        ax.set_title(filename_base)
        plt.plot(f[mean], p[mean], 'rv')
        plt.plot(f[mhm], p[mhm], 'gv')
        plt.plot(f[phm], p[phm], 'gv')

        # Now add the legend with some customizations.
        legend = ax.legend(loc='upper right', shadow=False)

        # Set legend fontsize
        for label in legend.get_texts():
            label.set_fontsize('small')

        plt.grid()
        plt.savefig('{}.png'.format(filename_wo_ext))
        print(filename_base, ' '.join(map(str, popt)), area)

        # Clear Canvases
        plt.clf()
        plt.cla()
        plt.close()


# ------------------------

if __name__ == '__main__':
    main()
