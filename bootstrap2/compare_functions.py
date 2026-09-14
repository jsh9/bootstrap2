# Copyright (c) 2016-present, Facebook, Inc.
# Copyright (c) 2026 jsh9
#
# SPDX-License-Identifier: MIT
#
# Based on the original bootstrapped library (Facebook BSD-licensed).
# See LICENSE for the MIT license and retained BSD attribution.
# An additional grant of patent rights can be found in the PATENTS file.
"""Various comparison functions for use in bootstrap a/b tests"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals


def difference(test_stat, ctrl_stat):
    """
    Calculates difference change. A good default. Args: test_stat: numpy array
    of test statistics ctrl_stat: numpy array of control statistics Returns:
    test_stat - ctrl_stat
    """
    return test_stat - ctrl_stat


def percent_change(test_stat, ctrl_stat):
    """
    Calculates percent change. Args: test_stat: numpy array of test statistics
    ctrl_stat: numpy array of control statistics Returns: (test_stat -
    ctrl_stat) / ctrl_stat * 100
    """
    return (test_stat - ctrl_stat) * 100.0 / abs(ctrl_stat)


def ratio(test_stat, ctrl_stat):
    """
    Calculates ratio between test and control Args: test_stat: numpy array of
    test statistics ctrl_stat: numpy array of control statistics Returns:
    test_stat / ctrl_stat
    """
    return test_stat / ctrl_stat


def percent_difference(test_stat, ctrl_stat):
    """
    Calculates ratio between test and control. Useful when your statistics
    might be close to zero. Provides a symmetric result. Args: test_stat: numpy
    array of test statistics ctrl_stat: numpy array of control statistics
    Returns: (test_stat - ctrl_stat) / ((test_stat + ctrl_stat) / 2.0) * 100.0
    """
    return (test_stat - ctrl_stat) / ((test_stat + ctrl_stat) / 2.0) * 100.0
