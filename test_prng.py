# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 francescoleonetti
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit test for the prng.
"""

import unittest
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
if sys.flags.interactive:
    plt.ion()

from prng import ProbabilityDensityFunction


class TestPdf(unittest.TestCase):

    def test_norm(self, xmin=0, xmax=1):

        x = np.linspace(xmin, xmax, 101)
        y = 2. / (xmax - xmin)**2. * (x - xmin)
        pdf = ProbabilityDensityFunction(x, y)

        norm = pdf.integral(xmin, xmax)
        self.assertAlmostEqual(norm, 1.0)

    def test_uniform(self):

        x = np.linspace(0., 1., 100)
        y = np.full(x.shape, 1.)
        pdf = ProbabilityDensityFunction(x, y)
        self.assertAlmostEqual(pdf(0.5), 1.)
        self.assertAlmostEqual(pdf.integral(0., 1.), 1.)
        self.assertAlmostEqual(pdf.prob(0.25, 0.75), 0.5)

    def test_triangular(self):
        x = np.linspace(0., 1., 100)
        y = 4. * x
        pdf = ProbabilityDensityFunction(x, y)
        plt.figure('Triangular pdf')
        plt.title('Triangular pdf')
        plt.plot(x, pdf(x), color='blue')
        plt.figure('Triangular cdf')
        plt.title('Triangular cdf')
        plt.plot(x, pdf.cdf(x), color='red')
        plt.figure('Triangular ppf')
        plt.title('Triangular ppf')
        plt.plot(x, pdf.ppf(x), color='darkorange')
        r = pdf.prng(1000000)
        plt.figure('Triangular random variate')
        plt.hist(r, bins=200, color='purple')

    def test_fancy(self):
        x = np.linspace(0., 1., 10000)
        y = np.zeros(x.shape)
        y[x <= 0.5] = 2. * x[x <= 0.5]
        y[x > 0.75] = 3
        pdf = ProbabilityDensityFunction(x, y, 1)
        plt.figure('Fancy pdf')
        plt.title('Fancy pdf')
        plt.plot(x, pdf(x), color='purple')
        print(pdf.integral(0., 1.))
        plt.figure('Fancy cdf')
        plt.title('Fancy cdf')
        plt.plot(x, pdf.cdf(x), color='brown')
        plt.figure('Fancy ppf')
        plt.title('Fancy ppf')
        plt.plot(x, pdf.ppf(x), color='black')
        r = pdf.prng(1000000)
        plt.figure('Fancy random variate')
        plt.title('Fancy random variate')
        plt.hist(r, bins=200, color='darkgreen')
        print(r)

    def test_fancy_bis(self):
        x = np.linspace(0., np.pi, 2000)
        y = np.sin((np.cos(np.sin(x)))**2)
        pdf = ProbabilityDensityFunction(x, y)
        plt.figure('Fancy pdf bis')
        plt.title('Fancy pdf bis')
        plt.plot(x, pdf(x), color='darkgreen')
        r = pdf.prng(1000000)
        print(r)
        plt.figure('Random numbers')
        plt.title('Random numbers')
        plt.hist(r, bins=200, color='purple')




if __name__ == '__main__':
    unittest.main(exit=not sys.flags.interactive)     


