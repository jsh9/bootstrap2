# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
# Copyright (c) 2026 jsh9
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Functions that allow one to create bootstrapped confidence intervals"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as _np
import multiprocessing as _multiprocessing
import scipy.sparse as _sparse


class BootstrapResults(object):
    def __init__(self, lower_bound, value, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.value = value
        if self.lower_bound > self.upper_bound:
            self.lower_bound, self.upper_bound = (
                self.upper_bound,
                self.lower_bound,
            )

    def __str__(self):
        return '{1}    ({0}, {2})'.format(
            self.lower_bound, self.value, self.upper_bound
        )

    def __repr__(self):
        return self.__str__()

    def _apply(self, other, func):
        return BootstrapResults(
            func(self.lower_bound, other),
            func(self.value, other),
            func(self.upper_bound, other),
        )

    def __add__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __radd__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __sub__(self, other):
        return self._apply(float(other), lambda x, other: x - other)

    def __rsub__(self, other):
        return self._apply(float(other), lambda x, other: other - x)

    def __mul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def __rmul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def error_width(self):
        """
        Return the width of the confidence interval.

        Returns
        -------
        float
            ``upper_bound - lower_bound``
        """
        return self.upper_bound - self.lower_bound

    def error_fraction(self):
        """
        Return the width of the confidence interval relative to the value.

        Returns
        -------
        float
            ``error_width() / value``, or ``numpy.inf`` if ``value`` is 0
        """
        if self.value == 0:
            return _np.inf
        else:
            return self.error_width() / self.value

    def is_significant(self):
        return _np.sign(self.upper_bound) == _np.sign(self.lower_bound)

    def get_result(self):
        """
        Return the direction of a statistically significant result.

        Returns
        -------
        float
            -1 if statistically significantly negative, +1 if statistically
            significantly positive, and 0 otherwise
        """
        return int(self.is_significant()) * _np.sign(self.value)


def _get_confidence_interval(bootstrap_dist, stat_val, alpha, is_pivotal):
    """
    Get the bootstrap confidence interval for a given distribution.

    Parameters
    ----------
    bootstrap_dist : numpy.ndarray
        Bootstrap results, such as those from ``_bootstrap_distribution()``
    stat_val : float
        The overall statistic that this method is attempting to calculate error
        bars for
    alpha : float
        The alpha value for the confidence interval
    is_pivotal : bool
        If True, use the pivotal method. If False, use the percentile method.

    Returns
    -------
    BootstrapResults
        The confidence interval and the estimated value
    """
    if is_pivotal:
        low = 2 * stat_val - _np.percentile(
            bootstrap_dist, 100 * (1 - alpha / 2.0)
        )
        val = stat_val
        high = 2 * stat_val - _np.percentile(
            bootstrap_dist, 100 * (alpha / 2.0)
        )
    else:
        low = _np.percentile(bootstrap_dist, 100 * (alpha / 2.0))
        val = _np.percentile(bootstrap_dist, 50)
        high = _np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.0))

    return BootstrapResults(low, val, high)


def _needs_sparse_unification(values_lists):
    non_zeros = values_lists[0] != 0

    for v in values_lists:
        v_nz = v != 0
        non_zeros = (non_zeros + v_nz) > 0

    non_zero_size = non_zeros.sum()

    for v in values_lists:
        if non_zero_size != v.data.shape[0]:
            return True

    return False


def _validate_arrays(values_lists):
    t = values_lists[0]
    t_type = type(t)
    if not isinstance(t, _sparse.csr_matrix) and not isinstance(
        t, _np.ndarray
    ):
        raise ValueError(
            (
                'The arrays must either be of type '
                'scipy.sparse.csr_matrix or numpy.array'
            )
        )

    for _, values in enumerate(values_lists[1:]):
        if not isinstance(values, t_type):
            raise ValueError('The arrays must all be of the same type')

        if t.shape != values.shape:
            raise ValueError('The arrays must all be of the same shape')

        if isinstance(t, _sparse.csr_matrix):
            if values.shape[0] > 1:
                raise ValueError(
                    ('The sparse matrix must have shape 1 row X N columns')
                )

    if isinstance(t, _sparse.csr_matrix):
        if _needs_sparse_unification(values_lists):
            raise ValueError(
                (
                    'The non-zero entries in the sparse arrays'
                    ' must be aligned: see '
                    'bootstrap2.unify_sparse_vectors function'
                )
            )


def _generate_distributions(values_lists, num_iterations):

    if isinstance(values_lists[0], _sparse.csr_matrix):
        # in the sparse case we dont actually need to bootstrap
        # the full sparse array since most values are 0
        # instead for each bootstrap iteration we:
        #    1. generate B number of non-zero entries to sample from the
        #          binomial distribution
        #    2. resample with replacement the non-zero entries from values
        #          B times
        #    3. create a new sparse array with the B resamples, zero otherwise
        results = [[] for _ in range(len(values_lists))]

        pop_size = values_lists[0].shape[1]
        non_sparse_size = values_lists[0].data.shape[0]

        p = non_sparse_size * 1.0 / pop_size

        for _ in range(num_iterations):
            ids = _np.random.choice(
                non_sparse_size,
                _np.random.binomial(pop_size, p),
                replace=True,
            )

            for arr, values in zip(results, values_lists):
                data = values.data
                d = _sparse.csr_matrix(
                    (data[ids], (_np.zeros_like(ids), _np.arange(len(ids)))),
                    shape=(1, pop_size),
                )

                arr.append(d)

        return [_sparse.vstack(r) for r in results]

    else:
        values_shape = values_lists[0].shape[0]
        ids = _np.random.choice(
            values_shape,
            (num_iterations, values_shape),
            replace=True,
        )

        results = [values[ids] for values in values_lists]
        return results


def _bootstrap_sim(
        values_lists,
        stat_func_lists,
        num_iterations,
        iteration_batch_size,
        seed,
):
    """
    Return a simulated bootstrap distribution.

    Parameters
    ----------
    values_lists : list[numpy.ndarray] or list[csr_matrix]
        The arrays to bootstrap. The same indices are resampled across all
        arrays.
    stat_func_lists : list[Callable]
        The statistic to apply to each array in ``values_lists``
    num_iterations : int
        The number of bootstrap iterations to run
    iteration_batch_size : int
        The number of iterations to simulate at a time
    seed : int or None
        The seed for NumPy's global random number generator. If None, the
        generator is not reseeded.

    Returns
    -------
    numpy.ndarray
        One row per array in ``values_lists`` and one column per bootstrap
        iteration
    """

    if seed is not None:
        _np.random.seed(seed)

    num_iterations = int(num_iterations)
    iteration_batch_size = int(iteration_batch_size)

    results = [[] for _ in values_lists]

    for rng in range(0, num_iterations, iteration_batch_size):
        max_rng = min(iteration_batch_size, num_iterations - rng)

        values_sims = _generate_distributions(values_lists, max_rng)

        for i, values_sim, stat_func in zip(
            range(len(values_sims)), values_sims, stat_func_lists
        ):
            results[i].extend(stat_func(values_sim))

    return _np.array(results)


def _bootstrap_distribution(
        values_lists,
        stat_func_lists,
        num_iterations,
        iteration_batch_size,
        num_threads,
):
    """
    Return the simulated bootstrap distribution.

    The idea is to sample the same indices in a bootstrap resample across all
    arrays passed into ``values_lists``. This is especially useful when you
    want to co-sample records in a ratio metric::

        numerator[k].sum() / denominator[k].sum()

    and not::

        numerator[j].sum() / denominator[k].sum()

    Parameters
    ----------
    values_lists : list[numpy.ndarray] or list[csr_matrix]
        Each element is a set of values to bootstrap. All arrays in
        ``values_lists`` must have the same length.
    stat_func_lists : list[Callable]
        The statistic to bootstrap for each element in ``values_lists``
    num_iterations : int
        The number of bootstrap iterations (resamples) to perform
    iteration_batch_size : int or None
        The bootstrap can generate very large matrices. This argument limits
        the memory footprint by batching bootstrap rounds: matrices of
        ``len(values) x iteration_batch_size`` are produced one at a time until
        ``num_iterations`` have been simulated. If None, a single matrix of
        ``len(values) x num_iterations`` is produced.
    num_threads : int
        The number of processes to use. If -1, ``multiprocessing.cpu_count()``
        is used.

    Returns
    -------
    numpy.ndarray
        The bootstrap distribution: one row per array in ``values_lists``,
        where each row holds the statistic computed on every resample
    """

    _validate_arrays(values_lists)

    if iteration_batch_size is None:
        iteration_batch_size = num_iterations

    num_iterations = int(num_iterations)
    iteration_batch_size = int(iteration_batch_size)

    num_threads = int(num_threads)

    if num_threads == -1:
        num_threads = _multiprocessing.cpu_count()

    if num_threads <= 1:
        results = _bootstrap_sim(
            values_lists,
            stat_func_lists,
            num_iterations,
            iteration_batch_size,
            None,
        )
    else:
        pool = _multiprocessing.Pool(num_threads)

        iter_per_job = _np.ceil(num_iterations * 1.0 / num_threads)

        results = []
        # dtype=int64: on Windows the default integer type is int32, and
        # 2**32 - 1 overflows int32 ("high is out of bounds for int32").
        for seed in _np.random.randint(
            0, 2**32 - 1, num_threads, dtype=_np.int64
        ):
            r = pool.apply_async(
                _bootstrap_sim,
                (
                    values_lists,
                    stat_func_lists,
                    iter_per_job,
                    iteration_batch_size,
                    seed,
                ),
            )
            results.append(r)

        results = _np.hstack([res.get() for res in results])

        pool.close()

    return results


def bootstrap(
        values,
        stat_func,
        denominator_values=None,
        alpha=0.05,
        num_iterations=10000,
        iteration_batch_size=10,
        is_pivotal=True,
        num_threads=1,
        return_distribution=False,
):
    """
    Return a bootstrap estimate and its confidence interval.

    Parameters
    ----------
    values : numpy.ndarray or csr_matrix
        The values to bootstrap, as a NumPy array or a
        ``scipy.sparse.csr_matrix`` with 1 row
    stat_func : Callable
        The statistic to bootstrap. It receives a 2D array with one resample
        per row and returns one value per row. Provided functions include
        ``stat_functions.mean``, ``stat_functions.sum``, and
        ``stat_functions.std``.
    denominator_values : numpy.ndarray or csr_matrix or None, default=None
        Optional array with one entry per record in ``values``. If given, the
        statistic is computed on ``values`` and on ``denominator_values``
        separately and then divided. This lets you compute group-level ratio
        statistics. For example::

            SUM(value) / SUM(denom) instead of MEAN(value / denom)
            Ex. Cost Per Click
            cost per click across a group
                SUM(revenue) / SUM(clicks)
            mean cost per click for each record
                MEAN(revenue / clicks)
    alpha : float, default=0.05
        The alpha value of the confidence interval. 0.05 gives a 95% confidence
        interval.
    num_iterations : int, default=10000
        The number of bootstrap iterations to run. The higher this number, the
        more stable the returned interval is across runs for the same input.
        More iterations also use more memory and make the analysis slower.
    iteration_batch_size : int or None, default=10
        The bootstrap can generate very large matrices. This argument limits
        the memory footprint by batching bootstrap rounds: matrices of
        ``len(values) x iteration_batch_size`` are produced one at a time until
        ``num_iterations`` have been simulated. If None, the full simulation is
        calculated in one step.
    is_pivotal : bool, default=True
        If True, use the pivotal method for bootstrapping confidence intervals.
        If False, use the percentile method.
    num_threads : int, default=1
        The number of processes to use to speed up the bootstrap. If -1,
        ``multiprocessing.cpu_count()`` is used.
    return_distribution : bool, default=False
        If True, return the bootstrap distribution instead of the confidence
        interval.

    Returns
    -------
    BootstrapResults or numpy.ndarray
        The confidence interval and estimated value, or the bootstrap
        distribution if ``return_distribution`` is True
    """
    if denominator_values is None:
        values_lists = [values]
        stat_func_lists = [stat_func]

        def do_division(x):
            return x

        stat_val = stat_func(values)[0]
    else:
        values_lists = [values, denominator_values]
        stat_func_lists = [stat_func] * 2

        def do_division(num, denom):
            return num / denom

        stat_val = stat_func(values)[0] / stat_func(denominator_values)[0]

    distribution_results = _bootstrap_distribution(
        values_lists,
        stat_func_lists,
        num_iterations,
        iteration_batch_size,
        num_threads,
    )

    bootstrap_dist = do_division(*distribution_results)

    if return_distribution:
        return bootstrap_dist
    else:
        return _get_confidence_interval(
            bootstrap_dist, stat_val, alpha, is_pivotal
        )


def bootstrap_ab(
        test,
        ctrl,
        stat_func,
        compare_func,
        test_denominator=None,
        ctrl_denominator=None,
        alpha=0.05,
        num_iterations=10000,
        iteration_batch_size=10,
        scale_test_by=1.0,
        is_pivotal=True,
        num_threads=1,
        return_distribution=False,
):
    """
    Return bootstrap confidence intervals for an A/B test.

    Parameters
    ----------
    test : numpy.ndarray or csr_matrix
        The test results, as a NumPy array or a ``scipy.sparse.csr_matrix``
        with 1 row
    ctrl : numpy.ndarray or csr_matrix
        The control results, as a NumPy array or a ``scipy.sparse.csr_matrix``
        with 1 row
    stat_func : Callable
        The statistic to bootstrap. It receives a 2D array with one resample
        per row and returns one value per row. Provided functions include
        ``stat_functions.mean``, ``stat_functions.sum``, and
        ``stat_functions.std``.
    compare_func : Callable
        The function that compares the test statistic against the control
        statistic. Provided functions include ``compare_functions.difference``,
        ``compare_functions.percent_change``, ``compare_functions.ratio``, and
        ``compare_functions.percent_difference``.
    test_denominator : numpy.ndarray or csr_matrix or None, default=None
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
    ctrl_denominator : numpy.ndarray or csr_matrix or None, default=None
        Optional array with one entry per record in ``ctrl``. See
        ``test_denominator``.
    alpha : float, default=0.05
        The alpha value of the confidence interval. 0.05 gives a 95% confidence
        interval.
    num_iterations : int, default=10000
        The number of bootstrap iterations to run. The higher this number, the
        more stable the returned interval is across runs for the same input.
        More iterations also use more memory and make the analysis slower.
    iteration_batch_size : int or None, default=10
        The bootstrap can generate very large arrays. This argument limits the
        memory footprint by batching bootstrap rounds. If None, the full
        simulation is calculated in one step.
    scale_test_by : float, default=1.0
        The ratio between the test and control population sizes. Use this if
        your test and control split is different from a 50/50 split.
    is_pivotal : bool, default=True
        If True, use the pivotal method for bootstrapping confidence intervals.
        If False, use the percentile method.
    num_threads : int, default=1
        The number of processes to use to speed up the bootstrap. If -1,
        ``multiprocessing.cpu_count()`` is used.
    return_distribution : bool, default=False
        If True, return the distribution of ``compare_func`` results instead of
        the confidence interval.

    Returns
    -------
    BootstrapResults or numpy.ndarray
        The confidence interval and estimated value, or the bootstrap
        distribution if ``return_distribution`` is True

    Raises
    ------
    ValueError
        If ``test`` or ``ctrl`` is None
    """

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

    test_results = _bootstrap_distribution(
        test_lists,
        stat_func_lists,
        num_iterations,
        iteration_batch_size,
        num_threads,
    )

    ctrl_results = _bootstrap_distribution(
        ctrl_lists,
        stat_func_lists,
        num_iterations,
        iteration_batch_size,
        num_threads,
    )

    test_dist = do_division(*test_results)
    ctrl_dist = do_division(*ctrl_results)

    test_ctrl_dist = compare_func(test_dist * scale_test_by, ctrl_dist)

    if return_distribution:
        return test_ctrl_dist
    else:
        test_ctrl_val = compare_func(test_val * scale_test_by, ctrl_val)
        return _get_confidence_interval(
            test_ctrl_dist, test_ctrl_val, alpha, is_pivotal
        )
