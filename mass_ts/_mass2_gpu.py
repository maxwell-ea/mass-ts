"""
© 2025 Emily Maxwell Outland <emily.maxwell@colorado.edu>
SPDX License: BSD-3-Clause

_mass2_gpu.py

Last Modified: 6-24-2025

Created this file, so the warnings about the gpu version not being usable stay contained to this function only.
"""

# From _mass_ts

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import warnings
import numpy as np
from mass_ts import core as mtscore

try:
    import cupy as cp
except ModuleNotFoundError:
    warnings.warn(
        'GPU support will not work: cupy is not installed.')


def mass2_gpu(ts, query):
    """
    Compute the distance profile for the given query over the given time
    series. This requires cupy to be installed.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.

    Returns
    -------
    An array of distances.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
    """

    # First, define this function:
    def moving_mean_std_gpu(a, w):
        s = cp.concatenate([cp.array([0]), cp.cumsum(a)])
        sSq = cp.concatenate([cp.array([0]), cp.cumsum(a ** 2)])
        segSum = s[w:] - s[:-w]
        segSumSq = sSq[w:] - sSq[:-w]

        movmean = segSum / w
        movstd = cp.sqrt(segSumSq / w - (segSum / w) ** 2)

        return (movmean, movstd)

    # Then, MASS2 GPU Version Code:
    x = cp.asarray(ts)
    y = cp.asarray(query)
    n = x.size
    m = y.size

    meany = cp.mean(y)
    sigmay = cp.std(y)

    meanx, sigmax = moving_mean_std_gpu(x, m)
    meanx = cp.concatenate([cp.ones(n - meanx.size), meanx])
    sigmax = cp.concatenate([cp.zeros(n - sigmax.size), sigmax])

    y = cp.concatenate((cp.flip(y, axis=0), cp.zeros(n - m)))

    X = cp.fft.fft(x)
    Y = cp.fft.fft(y)
    Z = X * Y
    z = cp.fft.ifft(Z)

    dist = 2 * (m - (z[m - 1:n] - m * meanx[m - 1:n] * meany) /
                (sigmax[m - 1:n] * sigmay))
    dist = cp.sqrt(dist)

    return cp.asnumpy(dist)
