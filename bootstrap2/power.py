# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
# Copyright (c) 2026 jsh9
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Functions that allow one to perform power analysis"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pandas as _pd
import numpy as _np
import warnings as _warnings


def _get_power_df(bootstrap_result_list):
    """
    Return a dataframe with important statistics for power analysis.

    Parameters
    ----------
    bootstrap_result_list : list[BootstrapResults]
        Results of many simulated bootstrap A/B tests. At least 3,000 results
        are recommended.

    Returns
    -------
    pandas.DataFrame
        One row per result, sorted by value, with significance and plotting
        columns added
    """
    if len(bootstrap_result_list) < 3000:
        _warnings.warn(
            (
                'bootstrap_result_list has very few examples. '
                'A general heuristic is to have at least 3k values. '
                'The more examples the more confident you can be in '
                'the power'
            )
        )

    df = _pd.DataFrame.from_dict([x.__dict__ for x in bootstrap_result_list])

    df = df.sort_values(by=['value']).reset_index().dropna()

    is_sig = df['upper_bound'].apply(_np.sign)
    is_sig = is_sig == df['lower_bound'].apply(_np.sign)
    df['is_significant'] = is_sig

    df['test_result'] = df['upper_bound'].apply(_np.sign).astype(int) * df[
        'is_significant'
    ].apply(lambda x: 1 if x else 0)

    result_cols = [
        'negative_significant',
        'insignificant',
        'positive_significant',
    ]
    df['test_result'] = df['test_result'].apply(lambda x: result_cols[x + 1])

    df['lower_bound_relative'] = df['value'] - df['lower_bound']
    df['upper_bound_relative'] = df['upper_bound'] - df['value']

    df['x_val'] = _np.arange(len(df))

    return df


def power_stats(bootstrap_result_list):
    """
    Return summary statistics about the power of a simulation.

    Parameters
    ----------
    bootstrap_result_list : list[BootstrapResults]
        Results of many simulated bootstrap A/B tests. At least 3,000 results
        are recommended.

    Returns
    -------
    pandas.DataFrame
        The percentage of results that are positive, negative, insignificant,
        positive significant, and negative significant

    Examples
    --------
    >>> import numpy as np
    >>> import bootstrap2.bootstrap as bs
    >>> import bootstrap2.compare_functions as bs_compare
    >>> import bootstrap2.power as bs_power
    >>> import bootstrap2.stats_functions as bs_stats
    >>> results = []
    >>> for _ in range(3000):
    ...     test = np.random.normal(loc=100, scale=100, size=500) * 1.05
    ...     ctrl = np.random.normal(loc=100, scale=10, size=500)
    ...     results.append(
    ...         bs.bootstrap_ab(
    ...             test, ctrl, bs_stats.mean, bs_compare.percent_change
    ...         )
    ...     )
    >>> bs_power.power_stats(results)  # doctest: +SKIP
    """
    power_df = _get_power_df(bootstrap_result_list)
    pcnt_results = power_df.test_result.value_counts() * 100 / len(power_df)

    sd = {
        'Positive': [(power_df.value > 0).mean() * 100],
        'Negative': [(power_df.value < 0).mean() * 100],
        'Insignificant': [pcnt_results.get('insignificant', 0)],
        'Positive Significant': [pcnt_results.get('positive_significant', 0)],
        'Negative Significant': [pcnt_results.get('negative_significant', 0)],
    }
    stats = _pd.DataFrame(sd).transpose()
    stats.columns = ['Percent']
    return stats


def plot_power(
        bootstrap_result_list,
        insignificant_color='blue',
        significant_color='orange',
        trend_color='black',
        zero_color='black',
):
    """
    Plot the results of a power simulation with matplotlib.

    Parameters
    ----------
    bootstrap_result_list : list[BootstrapResults]
        Results of many simulated bootstrap A/B tests. At least 3,000 results
        are recommended.
    insignificant_color : str, default='blue'
        The color of the error bars of insignificant results
    significant_color : str, default='orange'
        The color of the error bars of significant results
    trend_color : str, default='black'
        The color of the line connecting the estimated values
    zero_color : str, default='black'
        The color of the horizontal line at zero

    Examples
    --------
    >>> import numpy as np
    >>> import bootstrap2.bootstrap as bs
    >>> import bootstrap2.compare_functions as bs_compare
    >>> import bootstrap2.power as bs_power
    >>> import bootstrap2.stats_functions as bs_stats
    >>> results = []
    >>> for _ in range(3000):
    ...     test = np.random.normal(loc=100, scale=100, size=500) * 1.05
    ...     ctrl = np.random.normal(loc=100, scale=10, size=500)
    ...     results.append(
    ...         bs.bootstrap_ab(
    ...             test, ctrl, bs_stats.mean, bs_compare.percent_change
    ...         )
    ...     )
    >>> bs_power.plot_power(results)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    power_df = _get_power_df(bootstrap_result_list)
    sel = ~power_df['is_significant']

    plt.axhline(0, c=zero_color)

    plt.plot(power_df.x_val, power_df.value, c=trend_color)

    plt.errorbar(
        power_df[sel]['x_val'],
        power_df[sel]['value'],
        yerr=(
            power_df[sel]['lower_bound_relative'],
            power_df[sel]['upper_bound_relative'],
        ),
        c=insignificant_color,
        linestyle=' ',
    )

    plt.errorbar(
        power_df[~sel]['x_val'],
        power_df[~sel]['value'],
        yerr=(
            power_df[~sel]['lower_bound_relative'],
            power_df[~sel]['upper_bound_relative'],
        ),
        c=significant_color,
        linestyle=' ',
    )

    plt.xlim(0, len(power_df))
