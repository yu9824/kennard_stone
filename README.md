# Kennard Stone
![python_badge](https://img.shields.io/pypi/pyversions/kennard-stone)
![license_badge](https://img.shields.io/pypi/l/kennard-stone)
![PyPI_Downloads_badge](https://img.shields.io/pypi/dm/kennard-stone)
![Downloads](https://pepy.tech/badge/kennard-stone)

## What is this?
This is an algorithm for evenly partitioning data in a `scikit-learn`-like interface. (See [References](#References) for details of the algorithm.)

![simulateion_gif](https://github.com/yu9824/kennard_stone/blob/main/example/simulate.gif?raw=true "Simulateion")

## How to install
```bash
pip install kennard-stone
```
You need `numpy`, `pandas` and `scikit-learn`.

## How to use
You can use them like [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection).

See [example](https://github.com/yu9824/kennard_stone/tree/main/example) for details.

In the following, `X` denotes an arbitrary explanatory variable and `y` an arbitrary objective variable.
And, `estimator` indicates an arbitrary prediction model that conforms to scikit-learn.

### train_test_split
#### kennard_stone
```python
from kennard_stone import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```

#### scikit-learn
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 334)
```

### KFold
#### kennard_stone
```python
from kennard_stone import KFold

kf = KFold(n_splits = 5)
for i_train, i_test in kf.split(X, y):
    X_train = X[i_train]
    y_train = y[i_train]
    X_test = X[i_test]
    y_test = y[i_test]
```

#### scikit-learn
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 334)
for i_train, i_test in kf.split(X, y):
    X_train = X[i_train]
    y_train = y[i_train]
    X_test = X[i_test]
    y_test = y[i_test]
```

### Others
If you ever specify `cv` in scikit-learn, you can assign `KFold` objects to it and apply it to various functions.

An example is [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).

#### kennard_stone
```python
from kennard_stone import KFold
from sklearn.model_selection import cross_validate

kf = KFold(n_splits = 5)
print(cross_validate(estimator, X, y, cv = kf))
```

#### scikit-learn
```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

kf = KFold(n_splits = 5, shuffle = True, random_state = 334)
print(cross_validate(estimator, X, y, cv = kf))
```
OR
```python
from sklearn.model_selection import cross_validate

print(cross_validate(estimator, X, y, cv = 5))
```


## Points to note
There is no notion of `random_state` or `shuffle` because the partitioning is determined uniquely for the dataset.<br>
If you include them in the argument, you will not get an error, but they have no effect, so be careful.<br><br>
Also, `n_jobs` has not been implemented.


## References
### Papers
* R. W. Kennard & L. A. Stone (1969) Computer Aided Design of Experiments, Technometrics, 11:1, 137-148, DOI: [10.1080/00401706.1969.10490666](https://doi.org/10.1080/00401706.1969.10490666)
### Sites
* [https://datachemeng.com/trainingtestdivision/](https://datachemeng.com/trainingtestdivision/)