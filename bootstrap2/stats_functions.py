# Copyright (c) 2016-present, Facebook, Inc.
# Copyright (c) 2026 jsh9
#
# SPDX-License-Identifier: MIT
#
# Based on the original bootstrapped library (Facebook BSD-licensed).
# See LICENSE for the MIT license and retained BSD attribution.
# An additional grant of patent rights can be found in the PATENTS file.
"""Various comparison statistics functions to run on bootstrap simulations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as _np
import scipy.sparse as _sparse


def mean(values, axis=1):
    """Returns the mean of each row of a matrix"""
    if isinstance(values, _sparse.csr_matrix):
        ret = values.mean(axis=axis)
        return ret.A1
    else:
        return _np.mean(_np.asmatrix(values), axis=axis).A1


def sum(values, axis=1):
    """Returns the sum of each row of a matrix"""
    if isinstance(values, _sparse.csr_matrix):
        ret = values.sum(axis=axis)
        return ret.A1
    else:
        return _np.sum(_np.asmatrix(values), axis=axis).A1


def median(values, axis=1):
    """Returns the sum of each row of a matrix"""
    if isinstance(values, _sparse.csr_matrix):
        ret = values.median(axis=axis)
        return ret.A1
    else:
        return _np.median(_np.asmatrix(values), axis=axis).A1


def std(values, axis=1):
    """Returns the std of each row of a matrix"""
    if isinstance(values, _sparse.csr_matrix):
        ret = values.std(axis=axis)
        return ret.A1
    else:
        return _np.std(_np.asmatrix(values), axis=axis).A1
