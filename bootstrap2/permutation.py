# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
# Copyright (c) 2026 jsh9
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Functions that allow one to run a permutation shuffle test"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as _np
import multiprocessing as _multiprocessing
from warnings import warn

MAX_ITER = 10000
MAX_ARRAY_SIZE = 10000


# Randomized permutation shuffle test
def _get_permutation_result(permutation_dist, stat_val):
    """
    Get the permutation test result for a given distribution.

    Parameters
    ----------
    permutation_dist : numpy.ndarray
        Permutation shuffle results, such as those from
        ``_permutation_distribution()``
    stat_val : float
        The observed statistic to compare against the permutation distribution

    Returns
    -------
    float
        The two-sided p-value: the fraction of ``permutation_dist`` that is at
        least as far from 0 as ``stat_val``
    """

    denom = len(permutation_dist)

    pct = (
        len(permutation_dist[_np.where(permutation_dist >= abs(stat_val))])
        + len(permutation_dist[_np.where(permutation_dist <= -abs(stat_val))])
    ) / denom

    return pct


def _validate_arrays(values_lists):
    t = values_lists[0]
    t_type = type(t)
    if not isinstance(t, _np.ndarray):
        raise ValueError(('The arrays must be of type numpy.array'))

    for _, values in enumerate(values_lists[1:]):
        if not isinstance(values, t_type):
            raise ValueError('The arrays must all be of the same type')

        if t.shape != values.shape:
            raise ValueError('The arrays must all be of the same shape')


def _generate_distributions(values_lists, num_iterations=0):
    values_shape = values_lists[0].shape[0]
    ids = []

    for _ in range(num_iterations):
        ids.append(
            _np.random.choice(values_shape, values_shape, replace=False)
        )

    ids = _np.array(ids)

    results = [values[ids] for values in values_lists]
    return results


def _permutation_sim(
        test_lists,
        ctrl_lists,
        stat_func_lists,
        num_iterations,
        iteration_batch_size,
        seed,
):
    """
    Return simulated permutation distributions for test and control.

    Parameters
    ----------
    test_lists : list[numpy.ndarray]
        The test arrays to pool and shuffle
    ctrl_lists : list[numpy.ndarray]
        The control arrays to pool and shuffle
    stat_func_lists : list[Callable]
        The statistic to apply to each shuffled array
    num_iterations : int
        The number of permutation shuffles to run
    iteration_batch_size : int
        The number of shuffles to simulate at a time
    seed : int or None
        The seed for NumPy's global random number generator. If None, the
        generator is not reseeded.

    Returns
    -------
    test_results : numpy.ndarray
        The statistic computed on the test part of each shuffle, with one row
        per array in ``test_lists``
    ctrl_results : numpy.ndarray
        The statistic computed on the control part of each shuffle, with one
        row per array in ``ctrl_lists``
    """

    if seed is not None:
        _np.random.seed(seed)

    num_iterations = int(num_iterations)
    iteration_batch_size = int(iteration_batch_size)

    values_lists = []
    for i in range(len(test_lists)):
        values_lists.append(_np.append(test_lists[i], ctrl_lists[i]))

    test_results = [[] for _ in test_lists]
    ctrl_results = [[] for _ in ctrl_lists]
    test_sims = [[] for _ in test_lists]
    ctrl_sims = [[] for _ in ctrl_lists]

    for rng in range(0, num_iterations, iteration_batch_size):
        max_rng = min(iteration_batch_size, num_iterations - rng)

        values_sims = _generate_distributions(values_lists, max_rng)
        for i, result in enumerate(values_sims):
            for j in result:
                test_sims[i].append(j[0 : len(test_lists[0])])
                ctrl_sims[i].append(j[len(test_lists[0]) :])

        for i, test_sim_stat_func in enumerate(
            zip(test_sims, stat_func_lists)
        ):
            test_sim, stat_func = test_sim_stat_func
            test_results[i].extend(stat_func(test_sim))

        for i, ctrl_sim_stat_func in enumerate(
            zip(ctrl_sims, stat_func_lists)
        ):
            ctrl_sim, stat_func = ctrl_sim_stat_func
            ctrl_results[i].extend(stat_func(ctrl_sim))

    return _np.array(test_results), _np.array(ctrl_results)


def _permutation_distribution(
        test_lists,
        ctrl_lists,
        stat_func_lists,
        num_iterations,
        iteration_batch_size,
        num_threads,
):
    """
    Return the simulated permutation distribution.

    The idea is to use the same shuffled indices across all arrays passed into
    ``test_lists`` and ``ctrl_lists``. This is especially useful when you want
    to co-sample records in a ratio::

        numerator[k].sum() / denominator[k].sum()

    and not::

        numerator[j].sum() / denominator[k].sum()

    Parameters
    ----------
    test_lists : list[numpy.ndarray]
        Each element is a set of test values to shuffle. All arrays in
        ``test_lists`` must have the same length.
    ctrl_lists : list[numpy.ndarray]
        Each element is a set of control values to shuffle. All arrays in
        ``ctrl_lists`` must have the same length.
    stat_func_lists : list[Callable]
        The statistic to compute for each element in ``test_lists`` and
        ``ctrl_lists``
    num_iterations : int
        The number of permutation shuffles to perform
    iteration_batch_size : int or None
        The permutation shuffle can generate very large matrices. This argument
        limits the memory footprint by batching shuffle rounds: matrices of
        ``len(values) x iteration_batch_size`` are produced one at a time until
        ``num_iterations`` have been simulated. If None, a single matrix of
        ``len(values) x num_iterations`` is produced.
    num_threads : int
        The number of processes to use. If -1, ``multiprocessing.cpu_count()``
        is used.

    Returns
    -------
    test_results : numpy.ndarray
        The statistic computed on the test part of each shuffle
    ctrl_results : numpy.ndarray
        The statistic computed on the control part of each shuffle
    """
    _validate_arrays(test_lists)
    _validate_arrays(ctrl_lists)

    if iteration_batch_size is None:
        iteration_batch_size = num_iterations

    num_iterations = int(num_iterations)
    iteration_batch_size = int(iteration_batch_size)

    num_threads = int(num_threads)

    if num_threads == -1:
        num_threads = _multiprocessing.cpu_count()

    if num_threads <= 1:
        test_results, ctrl_results = _permutation_sim(
            test_lists,
            ctrl_lists,
            stat_func_lists,
            num_iterations,
            iteration_batch_size,
            None,
        )
    else:
        pool = _multiprocessing.Pool(num_threads)

        iter_per_job = _np.ceil(num_iterations * 1.0 / num_threads)

        test_results = []
        ctrl_results = []
        # dtype=int64: on Windows the default integer type is int32, and
        # 2**32 - 1 overflows int32 ("high is out of bounds for int32").
        for seed in _np.random.randint(
            0, 2**32 - 1, num_threads, dtype=_np.int64
        ):
            job_args = (
                test_lists,
                ctrl_lists,
                stat_func_lists,
                iter_per_job,
                iteration_batch_size,
                seed,
            )
            t, c = pool.apply_async(_permutation_sim, job_args)
            test_results.append(t)
            ctrl_results.append(c)

        test_results = _np.hstack([res.get() for res in test_results])
        ctrl_results = _np.hstack([res.get() for res in ctrl_results])

        pool.close()

    return test_results, ctrl_results


def permutation_test(
        test,
        ctrl,
        stat_func,
        compare_func,
        test_denominator=None,
        ctrl_denominator=None,
        num_iterations=10000,
        iteration_batch_size=None,
        num_threads=1,
        return_distribution=False,
):
    """
    Run a randomized permutation shuffle test for an A/B test.

    Parameters
    ----------
    test : numpy.ndarray
        The test results
    ctrl : numpy.ndarray
        The control results
    stat_func : Callable
        The statistic to compute on each shuffle. It receives a 2D array with
        one shuffle per row and returns one value per row. Provided functions
        include ``stat_functions.mean``, ``stat_functions.sum``, and
        ``stat_functions.std``.
    compare_func : Callable
        The function that compares the test statistic against the control
        statistic. Provided functions include ``compare_functions.difference``,
        ``compare_functions.percent_change``, ``compare_functions.ratio``, and
        ``compare_functions.percent_difference``.
    test_denominator : numpy.ndarray or None, default=None
        Optional array with one entry per record in ``test``. If given, the
        statistic is computed on ``test`` and on ``test_denominator``
        separately and then divided. This lets you compute group-level ratio
        statistics. For example::

            SUM(value) / SUM(denom) instead of MEAN(value / denom)
            Ex. Cost Per Click
            cost per click across a group  (clicks is denominator)
                SUM(revenue) / SUM(clicks)
            mean cost per click for each record
                MEAN(revenue / clicks)
    ctrl_denominator : numpy.ndarray or None, default=None
        Optional array with one entry per record in ``ctrl``. See
        ``test_denominator``.
    num_iterations : int, default=10000
        The number of permutation shuffles to run. The higher this number, the
        more stable the result is across runs for the same input. More
        iterations also use more memory and make the analysis slower. If
        ``test`` or ``ctrl`` has at least ``MAX_ARRAY_SIZE`` elements, this is
        capped at ``MAX_ITER`` with a warning.
    iteration_batch_size : int or None, default=None
        The permutation shuffle can generate very large arrays. This argument
        limits the memory footprint by batching shuffle rounds. If None, the
        full simulation is calculated in one step.
    num_threads : int, default=1
        The number of processes to use to speed up the shuffle. If -1,
        ``multiprocessing.cpu_count()`` is used.
    return_distribution : bool, default=False
        If True, return the distribution of ``compare_func`` results instead of
        the p-value.

    Returns
    -------
    float or numpy.ndarray
        The fraction of permutation distribution values that are more extreme
        than the observed value (a two-sided p-value), or the permutation
        distribution if ``return_distribution`` is True

    Raises
    ------
    ValueError
        If ``test`` or ``ctrl`` is None
    """
    is_large_array = len(test) >= MAX_ARRAY_SIZE or len(ctrl) >= MAX_ARRAY_SIZE
    if is_large_array and num_iterations > MAX_ITER:
        warning_text = (
            'Maximum array length of {} exceeded, '
            'limiting num_iterations to {}'
        )
        warn(warning_text.format(MAX_ARRAY_SIZE, MAX_ITER))
        num_iterations = MAX_ITER

    both_denominators = (
        test_denominator is not None and ctrl_denominator is not None
    )
    both_numerators = test is not None and ctrl is not None

    if both_numerators and not both_denominators:
        test_lists = [test]
        ctrl_lists = [ctrl]
        stat_func_lists = [stat_func]

        def do_division(x):
            return x

        test_val = stat_func(test)[0]
        ctrl_val = stat_func(ctrl)[0]

    elif both_numerators and both_denominators:
        test_lists = [test, test_denominator]
        ctrl_lists = [ctrl, ctrl_denominator]
        stat_func_lists = [stat_func] * 2

        def do_division(num, denom):
            return num / denom

        test_val = stat_func(test)[0] / stat_func(test_denominator)[0]
        ctrl_val = stat_func(ctrl)[0] / stat_func(ctrl_denominator)[0]

    elif not both_numerators:
        raise ValueError('Both test and ctrl numerators must be specified.')
    else:
        raise ValueError('Both test and ctrl denominators must be specified.')

    test_results, ctrl_results = _permutation_distribution(
        test_lists,
        ctrl_lists,
        stat_func_lists,
        num_iterations,
        iteration_batch_size,
        num_threads,
    )

    test_dist = do_division(*test_results)
    ctrl_dist = do_division(*ctrl_results)

    test_ctrl_dist = compare_func(test_dist, ctrl_dist)

    if return_distribution:
        return test_ctrl_dist
    else:
        test_ctrl_val = compare_func(test_val, ctrl_val)
        return _get_permutation_result(test_ctrl_dist, test_ctrl_val)
