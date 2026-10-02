# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
# Copyright (c) 2026 jsh9
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Various comparison statistics functions to run on bootstrap simulations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as _np
import scipy.sparse as _sparse


def mean(values, axis=1):
    """
    Return the mean of each row of a matrix.

    Parameters
    ----------
    values : numpy.ndarray or csr_matrix
        A 2D array (or ``scipy.sparse.csr_matrix``) of values, where each row
        is one bootstrap resample
    axis : int, default=1
        The axis along which to compute the mean

    Returns
    -------
    numpy.ndarray
        A 1D array with the mean of each row (when ``axis`` is 1)
    """
    if isinstance(values, _sparse.csr_matrix):
        ret = values.mean(axis=axis)
        return ret.A1
    else:
        return _np.mean(_np.asmatrix(values), axis=axis).A1


def sum(values, axis=1):
    """
    Return the sum of each row of a matrix.

    Parameters
    ----------
    values : numpy.ndarray or csr_matrix
        A 2D array (or ``scipy.sparse.csr_matrix``) of values, where each row
        is one bootstrap resample
    axis : int, default=1
        The axis along which to compute the sum

    Returns
    -------
    numpy.ndarray
        A 1D array with the sum of each row (when ``axis`` is 1)
    """
    if isinstance(values, _sparse.csr_matrix):
        ret = values.sum(axis=axis)
        return ret.A1
    else:
        return _np.sum(_np.asmatrix(values), axis=axis).A1


def median(values, axis=1):
    """
    Return the median of each row of a matrix.

    Parameters
    ----------
    values : numpy.ndarray
        A 2D array of values, where each row is one bootstrap resample.
        ``scipy.sparse.csr_matrix`` input is not supported.
    axis : int, default=1
        The axis along which to compute the median

    Returns
    -------
    numpy.ndarray
        A 1D array with the median of each row (when ``axis`` is 1)
    """
    if isinstance(values, _sparse.csr_matrix):
        ret = values.median(axis=axis)
        return ret.A1
    else:
        return _np.median(_np.asmatrix(values), axis=axis).A1


def std(values, axis=1):
    """
    Return the standard deviation of each row of a matrix.

    Parameters
    ----------
    values : numpy.ndarray
        A 2D array of values, where each row is one bootstrap resample.
        ``scipy.sparse.csr_matrix`` input is not supported.
    axis : int, default=1
        The axis along which to compute the standard deviation

    Returns
    -------
    numpy.ndarray
        A 1D array with the standard deviation of each row (when ``axis`` is 1)
    """
    if isinstance(values, _sparse.csr_matrix):
        ret = values.std(axis=axis)
        return ret.A1
    else:
        return _np.std(_np.asmatrix(values), axis=axis).A1
