# Changelog

Changes since v3.0.0 are listed in the [GitHub Releases](https://github.com/yu9824/kennard_stone/releases).
The history below was kept in the README until v4.1.0.

## v2.0.0 (deprecated)

- Define Extended Kennard-Stone algorithm (multi-class) i.e. Improve KFold algorithm.
- Delete `alternate` argument in `KFold`.
- Delete requirements of `pandas`.

## v2.0.1

- Fix bug with Python3.7.

## v2.1.0 (deprecated)

- Optimize algorithm
- Deal with Large number of data.
  - parallel calculation when calculating distance (Add `n_jobs` argument)
  - replacing recursive functions with for-loops
- Add other than "euclidean" calculation methods (Add `metric` argument)

## v2.1.1 (deprecated)

- Fix bug when `metric="nan_euclidean"`.

## v2.1.2 (deprecated)

- Fix details.
  - Update docstrings and typings.

## v2.1.3 (deprecated)

- Fix details.
  - Update some typings. (You have access to a list of strings that can be used in the metric.)

## v2.1.4

- Fix bug when metric=="seuclidean" and "mahalanobis"
  - Add some tests to check all metrics.
- Add requirements numpy>=1.20

## v2.1.5

- Delete "klusinski" metric to support scipy>=1.11

## v2.1.6

- Improve typing in `kennard_stone.train_test_split`
- Add some docstrings.

## v2.2.0

- Supports GPU calculations. (when metric is 'euclidean', 'manhattan', 'chebyshev' and 'minkowski')
- Supports Python 3.12


## v2.2.1

- Fix setup.cfg
- Update 'typing'
