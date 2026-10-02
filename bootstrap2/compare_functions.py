# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
# Copyright (c) 2026 jsh9
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Various comparison functions for use in bootstrap a/b tests"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals


def difference(test_stat, ctrl_stat):
    """
    Calculate the difference between test and control. A good default.

    Parameters
    ----------
    test_stat : numpy.ndarray or float
        The test statistics
    ctrl_stat : numpy.ndarray or float
        The control statistics

    Returns
    -------
    numpy.ndarray or float
        ``test_stat - ctrl_stat``
    """
    return test_stat - ctrl_stat


def percent_change(test_stat, ctrl_stat):
    """
    Calculate the percent change from control to test.

    Parameters
    ----------
    test_stat : numpy.ndarray or float
        The test statistics
    ctrl_stat : numpy.ndarray or float
        The control statistics

    Returns
    -------
    numpy.ndarray or float
        ``(test_stat - ctrl_stat) * 100 / abs(ctrl_stat)``
    """
    return (test_stat - ctrl_stat) * 100.0 / abs(ctrl_stat)


def ratio(test_stat, ctrl_stat):
    """
    Calculate the ratio between test and control.

    Parameters
    ----------
    test_stat : numpy.ndarray or float
        The test statistics
    ctrl_stat : numpy.ndarray or float
        The control statistics

    Returns
    -------
    numpy.ndarray or float
        ``test_stat / ctrl_stat``
    """
    return test_stat / ctrl_stat


def percent_difference(test_stat, ctrl_stat):
    """
    Calculate the percent difference between test and control.

    This is useful when your statistics might be close to zero. It provides a
    symmetric result.

    Parameters
    ----------
    test_stat : numpy.ndarray or float
        The test statistics
    ctrl_stat : numpy.ndarray or float
        The control statistics

    Returns
    -------
    numpy.ndarray or float
        ``(test_stat - ctrl_stat) / ((test_stat + ctrl_stat) / 2.0) * 100.0``
    """
    return (test_stat - ctrl_stat) / ((test_stat + ctrl_stat) / 2.0) * 100.0
