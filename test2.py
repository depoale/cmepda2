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


class PdfTests(unittest.TestCase):
    def test_linear(self, x_min, x_max, slope):
        x = np.linspace(x_min, x_max, 100)
        y = x*slope
        pdf = ProbabilityDensityFunction(x, y)

        self.assertAlmostEqual(pdf.integral(x_min, x_max), 1.)


    def test_funct(self):
        self.test_linear(0., 1., 4)
        self.test_linear(0., 3., 1)

if __name__ == '__main__':
    unittest.main(exit=not sys.flags.interactive)
