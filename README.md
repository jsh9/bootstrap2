# bootstrap2 - confidence intervals made easy

**bootstrap2** is a Python library that allows you to build confidence
intervals from data. This is useful in a variety of contexts — including during
ad-hoc A/B test analysis.

It is a maintained continuation of the original
[bootstrapped](https://github.com/facebookincubator/bootstrapped) library
(renamed and refreshed for modern Python packaging).

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [1. Motivating example - A/B test](#1-motivating-example---ab-test)
- [2. The gist - mean of a sample](#2-the-gist---mean-of-a-sample)
- [3. Benefits](#3-benefits)
- [4. Example usage](#4-example-usage)
  - [4.1. Extended examples](#41-extended-examples)
- [5. Requirements](#5-requirements)
- [6. Installation](#6-installation)
- [7. How bootstrap2 works](#7-how-bootstrap2-works)
- [8. Contributors](#8-contributors)
- [9. License](#9-license)

______________________________________________________________________

<!--TOC-->

## 1. Motivating example - A/B test

Imagine we own a website and think changing the color of a 'subscribe' button
will improve signups. One method to measure the improvement is to conduct an
A/B test where we show 50% of people the old version and 50% of the people the
new version. We can use the bootstrap to understand how much the button color
improves responses and give us the error bars associated with the test — this
will give us lower and upper bounds on how good we should expect the change to
be!

## 2. The gist - mean of a sample

Given a sample of data, we can generate a bunch of new samples by 're-sampling'
from what we have gathered. We calculate the mean for each generated sample. We
can use the means from the generated samples to understand the variation in the
larger population and can construct error bars for the true mean.

## 3. Benefits

- Efficient computation of confidence intervals
- Functions to handle single populations and A/B tests
- Functions to understand
  [statistical power](https://en.wikipedia.org/wiki/Statistical_power)
- Multithreaded support to speed up bootstrap computations
- Dense and sparse array support

## 4. Example usage

```python
import numpy as np
import bootstrap2.bootstrap as bs
import bootstrap2.stats_functions as bs_stats

mean = 100
stdev = 10

population = np.random.normal(loc=mean, scale=stdev, size=50000)

# take 1k 'samples' from the larger population
samples = population[:1000]

print(bs.bootstrap(samples, stat_func=bs_stats.mean))
# (99.46, 100.69)

print(bs.bootstrap(samples, stat_func=bs_stats.std))
# (9.92, 10.36)
```

### 4.1. Extended examples

- [Bootstrap Intro](https://github.com/jsh9/bootstrap2/blob/main/examples/bootstrap_intro.ipynb)
- [Bootstrap A/B Testing](https://github.com/jsh9/bootstrap2/blob/main/examples/bootstrap_ab_testing.ipynb)
- More notebooks can be found in the
  [examples/](https://github.com/jsh9/bootstrap2/tree/main/examples) directory

## 5. Requirements

**bootstrap2** requires Python 3.10+, numpy, scipy, matplotlib, and pandas.

## 6. Installation

```bash
pip install bootstrap2
```

## 7. How bootstrap2 works

**bootstrap2** provides pivotal (aka empirical) based confidence intervals
based on bootstrap re-sampling with replacement. The percentile method is also
available.

For more information please see:

1. [Bootstrap confidence intervals](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf)
   (good intro)
2. [An introduction to Bootstrap Methods](http://www.stat-athens.aueb.gr/~karlis/lefkada/boot.pdf)
3. [The Bootstrap, Advanced Data Analysis](http://www.stat.cmu.edu/~cshalizi/402/lectures/08-bootstrap/lecture-08.pdf)
4. [When the bootstrap doesn't work](http://notstatschat.tumblr.com/post/156650638586/when-the-bootstrap-doesnt-work)
5. (book)
   [An Introduction to the Bootstrap](https://www.amazon.com/Introduction-Bootstrap-Monographs-Statistics-Probability/dp/0412042312/)
6. (book)
   [Bootstrap Methods and their Application](https://www.amazon.com/Bootstrap-Application-Statistical-Probabilistic-Mathematics-ebook/dp/B00D2WQ02U/)

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help out.

## 8. Contributors

Spencer Beecher, Don van der Drift, David Martin, Lindsay Vass, Sergey Goder,
Benedict Lim, and Matt Langner.

Special thanks to Eytan Bakshy.

## 9. License

**bootstrap2** is MIT-licensed. The original Facebook BSD license for
`bootstrapped` is retained in [LICENSE-BSD](LICENSE-BSD) and cited in
[LICENSE](LICENSE). An additional patent grant from the original project is in
[PATENTS](PATENTS).
