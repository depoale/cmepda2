# Copyright (C) 2022 Alessia De Ponti
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import random
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


class ProbabilityDensityFunction(InterpolatedUnivariateSpline):

    """Class describing a probability density function.
    Parameters
    ----------
    x : array-like
        The array of x values to be passed to the pdf, assumed to be ordinate.
    y : array-like
        The array of y values to be passed to the pdf.
    """

    def __init__(self, x, y, k=3):
        """Constructor.
        """
        super().__init__(x, y, k)

        norm = InterpolatedUnivariateSpline(x, y, k).integral(x[0], x[-1])
        y /= norm
        f = InterpolatedUnivariateSpline(x, y, k)
        ycdf = np.array([f.integral(x[0], xcdf) for xcdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, ycdf)

        # scipy assume che non ci siano valori identici della x (x spline)
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf]
        self.ppf = InterpolatedUnivariateSpline(xppf, yppf)

    def prob(self, x1, x2):
        """Return the probability for the random variable to be included
        between x1 and x2.
        Parameters
        ----------
        x1: float or array-like
            The left bound for the integration.
        x2: float or array-like
            The right bound for the integration.
        """
        return self.cdf(x2) - self.cdf(x1)

    def rnd(self, size=1000):
        """Return an array of random values from the pdf.
        Parameters
        ----------
        size: int
            The number of random numbers to extract.
        """
        return self.ppf(np.random.uniform(size=size))


if __name__ == '__main__':
    pass
