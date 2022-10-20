# Copyright (C) 2022 Alessia De Ponti e Luca Baldini
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


from matplotlib import pyplot as plt
import numpy as np
import scipy.special


def exponential(x, mean):
    """Pure-python implementation of an exponential distribution.

    Arguments
    ---------
    x : array_like
        The independent variable.

    mean : float
        The mean value of the distribution.
    """
    return 1. / mean * np.exp(-x / mean)


def polya(x, mean, theta):
    """Pure-python implementation of a Polya distribution.

    Mind there are several different parametrization floating around in the
    literature. This particular one reduces to a simple exponantial for
    theta = 1.

    Arguments
    ---------
    x : array_like
        The independent variable.

    mean : float
        The mean value of the distribution.

    theta : float
        The Polya index.
    """
    norm = theta**theta / scipy.special.gamma(theta) / (mean**theta)
    return  norm * x**(theta - 1) *np.exp(-theta * x / mean)


def polya_variate(mean, theta, size=1):
    """Extract a numpy array of Polya variates.

    This relies on the function that the inverse to the regularized lower
    incomplete gamma function is the percent point function for the Polya
    distribution.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincinv.html#scipy.special.gammaincinv

    Arguments
    ---------
    mean : float
        The mean of the Polya variate.

    theta : float
        The theta parameter of the Polya variate

    size : int
        The number of values to be extracted.
    """
    u = np.random.random(size=size)
    return mean / theta * scipy.special.gammaincinv(theta, u)



if __name__ == '__main__':
    G = 20.
    x = np.linspace(0., 100., 500)

    plt.figure('Polya function')
    for theta in [1., 2., 5.]:
        plt.plot(x, polya(x, G, theta), label=f'$G = {G}$, $\\theta = {theta}$')
    plt.plot(x, exponential(x, G), ls='dashed', label='Exponential')
    plt.legend()

    plt.figure('Polya variate')
    N = 100000
    theta = 2.
    plt.plot(x, polya(x, G, theta), label='Polya function')
    v = polya_variate(G, theta, N)
    plt.hist(v, bins=200, density=True, label='Polya variate')

    plt.show()
