''' Polya sampling '''

import numpy as np
import random
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import time

from utils import polya
from pdf import ProbabilityDensityFunction


def random_polya(x, mean, theta, size):
    '''Generate random numbers with Polya distribution,
       return a list with the generated numbers '''

    y = polya(x, mean, theta)
    pdf = ProbabilityDensityFunction(x, y)

    return pdf.rnd(size=size)


if __name__ == '__main__':
    G = 20.
    theta = 3.
    num_events = 10000
    t0 = time.time()

    xx = np.linspace(0., 100., 200)
    plt.hist(random_polya(xx, G, theta, size=num_events), bins=100)
    print(f'Elapsed time: {time.time() - t0} seconds')
    plt.show()