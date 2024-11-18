# Kennard Stone

[![python_badge](https://img.shields.io/pypi/pyversions/kennard-stone)](https://pypi.org/project/kennard-stone/)
[![license_badge](https://img.shields.io/pypi/l/kennard-stone)](https://pypi.org/project/kennard-stone/)
[![PyPI version](https://badge.fury.io/py/kennard-stone.svg)](https://pypi.org/project/kennard-stone/)
[![Downloads](https://static.pepy.tech/badge/kennard-stone)](https://pepy.tech/project/kennard-stone)

[![Test on each version](https://github.com/yu9824/kennard_stone/actions/workflows/pytest-on-each-version.yaml/badge.svg)](https://github.com/yu9824/kennard_stone/actions/workflows/pytest-on-each-version.yaml)
[![docs](https://github.com/yu9824/kennard_stone/actions/workflows/docs.yml/badge.svg)](https://github.com/yu9824/kennard_stone/actions/workflows/docs.yml)
[![release-pypi](https://github.com/yu9824/kennard_stone/actions/workflows/release-pypi.yml/badge.svg)](https://github.com/yu9824/kennard_stone/actions/workflows/release-pypi.yml)

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/kennard-stone.svg)](https://anaconda.org/conda-forge/kennard-stone)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/kennard-stone.svg)](https://anaconda.org/conda-forge/kennard-stone)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)

## What is this?

This is an algorithm for evenly partitioning data in a `scikit-learn`-like interface.
(See [References](#References) for details of the algorithm.)

![simulation_gif](https://github.com/yu9824/kennard_stone/blob/main/examples/simulate.gif?raw=true "simulation_gif")

## How to install

### PyPI

```bash
pip install kennard-stone
```

The project site is [here](https://pypi.org/project/kennard-stone/).

### Anaconda

```bash
conda install -c conda-forge kennard-stone
```

The project site is [here](https://anaconda.org/conda-forge/kennard-stone).

You need `numpy>=1.20` and `scikit-learn` to run.

## How to use

You can use them like [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection).

See [examples](https://github.com/yu9824/kennard_stone/tree/main/examples) for details.

In the following, `X` denotes an arbitrary explanatory variable and `y` an arbitrary objective variable.
And, `estimator` indicates an arbitrary prediction model that conforms to scikit-learn.

### train_test_split

#### kennard_stone

```python
from kennard_stone import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

#### scikit-learn

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=334
)
```

### KFold

#### kennard_stone

```python
from kennard_stone import KFold

# Always shuffled and uniquely determined for a data set.
kf = KFold(n_splits=5)
for i_train, i_test in kf.split(X, y):
    X_train = X[i_train]
    y_train = y[i_train]
    X_test = X[i_test]
    y_test = y[i_test]
```

#### scikit-learn

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=334)
for i_train, i_test in kf.split(X, y):
    X_train = X[i_train]
    y_train = y[i_train]
    X_test = X[i_test]
    y_test = y[i_test]
```

### Other usages

If you ever specify `cv` in scikit-learn, you can assign `KFold` objects to it and apply it to various functions.

An example is [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).

#### kennard_stone

```python
from kennard_stone import KFold
from sklearn.model_selection import cross_validate

kf = KFold(n_splits=5)
print(cross_validate(estimator, X, y, cv=kf))
```

#### scikit-learn

```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

kf = KFold(n_splits=5, shuffle=True, random_state=334)
print(cross_validate(estimator, X, y, cv=kf))
```

OR

```python
from sklearn.model_selection import cross_validate

print(cross_validate(estimator, X, y, cv=5))
```


## Notes

There is no notion of `random_state` or `shuffle` because the partitioning is determined uniquely for the dataset.
If these arguments are included, they do not cause an error. They simply have no effect on the result. Please be careful.

If you want to run the notebook in examples directory,
you will need to additionally install `pandas`, `matplotlib`, `seaborn`, `tqdm`, and `jupyter` other than the packages in requirements.txt.

## Distance metrics

See the documentation of

- `scipy.spatial.distance.pdist`
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
- `sklearn.metrics.pairwise_distances`
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

Valid values for metric are:

- From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1',
    'l2', 'manhattan']. These metrics support sparse matrix inputs.
    ['nan_euclidean'] but it does not yet support sparse matrices.
- From scipy.spatial.distance: ['braycurtis', 'canberra',
    'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
    'mahalanobis', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
    'sqeuclidean', 'yule'] See the documentation for
    scipy.spatial.distance for details on these metrics.
    These metrics do not support sparse matrix inputs.

, by default "euclidean"

## Parallelization (since v2.1.0)

This algorithm is very computationally intensive and takes a lot of time.
To solve this problem, I have implemented parallelization and optimized the algorithm since v2.1.0.
`n_jobs` can be specified for parallelization as in the scikit-learn-like api.

```python
# parallelization KFold
kf = KFold(n_splits=5, n_jobs=-1)

# parallelization train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, n_jobs=-1
)
```

The parallelization is used when calculating the distance matrix,
so it doesn't conflict with something like `cross_validate`  in parallel when using `KFold`.

```python
# OK: does not conflict each other
cross_validate(estimator, X, y, cv=KFold(5, n_jobs=-1), n_jobs=-1)
```


## Using GPU

If you have a GPU and have installed pytorch,
you can use it to calculate Minkowski distances (Manhattan, Euclidean, and Chebyshev distances).


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, device="cuda"
)

```


## LICENSE

MIT Licence

Copyright (c) 2021 yu9824


## References

### Papers

- R. W. Kennard & L. A. Stone (1969) Computer Aided Design of Experiments, Technometrics, 11:1, 137-148, DOI: [10.1080/00401706.1969.10490666](https://doi.org/10.1080/00401706.1969.10490666)

### Sites

- [https://datachemeng.com/trainingtestdivision/](https://datachemeng.com/trainingtestdivision/) (Japanese site)


## Histories

### v2.0.0 (deprecated)

- Define Extended Kennard-Stone algorithm (multi-class) i.e. Improve KFold algorithm.
- Delete `alternate` argument in `KFold`.
- Delete requirements of `pandas`.

### v2.0.1

- Fix bug with Python3.7.

### v2.1.0 (deprecated)

- Optimize algorithm
- Deal with Large number of data.
  - parallel calculation when calculating distance (Add `n_jobs` argument)
  - replacing recursive functions with for-loops
- Add other than "euclidean" calculation methods (Add `metric` argument)

### v2.1.1 (deprecated)

- Fix bug when `metric="nan_euclidean"`.

### v2.1.2 (deprecated)

- Fix details.
  - Update docstrings and typings.

### v2.1.3 (deprecated)

- Fix details.
  - Update some typings. (You have access to a list of strings that can be used in the metric.)

### v2.1.4

- Fix bug when metric=="seuclidean" and "mahalanobis"
  - Add some tests to check all metrics.
- Add requirements numpy>=1.20

### v2.1.5

- Delete "klusinski" metric to support scipy>=1.11

### v2.1.6

- Improve typing in `kennard_stone.train_test_split`
- Add some docstrings.

### v2.2.0

- Supports GPU calculations. (when metric is 'euclidean', 'manhattan', 'chebyshev' and 'minkowski')
- Supports Python 3.12


### v2.2.1

- Fix setup.cfg
- Update 'typing'
