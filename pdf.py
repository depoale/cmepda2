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

'''
Module: basic Python
Assignment #4 (October 7, 2021)


--- Goal
Create a ProbabilityDensityFunction class that is capable of throwing
preudo-random number with an arbitrary distribution.

(In practice, start with something easy, like a triangular distribution---the
initial debug will be easier if you know exactly what to expect.)


--- Specifications
- the signature of the constructor should be __init__(self, x, y), where
  x and y are two numpy arrays sampling the pdf on a grid of values, that
  you will use to build a spline
- [optional] add more arguments to the constructor to control the creation
  of the spline (e.g., its order)
- the class should be able to evaluate itself on a generic point or array of
  points
- the class should be able to calculate the probability for the random
  variable to be included in a generic interval
- the class should be able to throw random numbers according to the distribution
  that it represents
- [optional] how many random numbers do you have to throw to hit the
  numerical inaccuracy of your generator?
'''
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
        # Normalize the pdf, if it is not (and probably it is npt!)
        norm = InterpolatedUnivariateSpline(x, y, k=k).integral(x[0], x[-1])
        y /= norm
        super().__init__(x, y, k=k)
        ycdf = np.array([self.integral(x[0], xcdf) for xcdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, ycdf, k=k)
        # Need to make sure that the vector I am passing to the ppf spline as
        # the x values (here we are interchanging x and y: the ppf is the inverse
        # function of the cdf!) has no duplicates---and need to filter the y
        # accordingly! We can only invert bijective functions!
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf]
        self.ppf = InterpolatedUnivariateSpline(xppf, yppf, k=k)

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
