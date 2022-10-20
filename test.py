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

''' Test '''

import unittest
import sys
import numpy as np
import random
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from pdf import ProbabilityDensityFunction
if sys.flags.interactive:
    plt.ion() #used to turn on interactive mode (plot blocca tutto)


#class PdfTests(unittest.TestCase):
    """Unit test for cdf"""
'''
    def test_uniform(self):
        """ """
        x = np.linspace(0., 1., 100)
        y = np.full(x.shape, 1.)
        pdf = ProbabilityDensityFunction(x, y)

        self.assertAlmostEqual(pdf(0.5), 1.)
        self.assertAlmostEqual(pdf.integral(0., 1.), 1.)
        self.assertAlmostEqual(pdf.prob(0.25, 0.75), 0.5)

    def test_triangular(self):
        x = np.linspace(0., 1., 100)
        y = 2*x
        pdf = ProbabilityDensityFunction(x, y)

    def test_fancy(self):
        """ """
        x = np.linspace(0., 1., 100)
        y = np.zeros(x.shape)
        y[x <= 0.5] = 2 * x[x <= 0.5]
        y[x > 0.75] = 3.
        self.assertEqual(len(x), len(y))
        pdf = ProbabilityDensityFunction(x, y, 1)
'''
class PdfTests(unittest.TestCase):
    def test_linear(self, x_min, x_max, slope):
        """ """

        x = np.linspace(x_min, x_max, 20)
        y = slope * x
        f = ProbabilityDensityFunction(x, y)

        #check normalization
        norm = f.integral(x_min, x_max)
        self.assertAlmostEqual(norm, 1.0)

        plt.hist(f.rnd(size=100000), bins=100)
        plt.show()

    def test__(self):
        self.test_linear(0., 2., 1.)
        self.test_linear(1., 4., 0.5)
        self.test_linear(1., 2., 3.)


if __name__ == '__main__':
    unittest.main(exit=not sys.flags.interactive)
