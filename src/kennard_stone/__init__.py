"""kennard_stone package
=========================

Utilities for splitting data as uniformly as possible using the
Kennard–Stone algorithm. The package exposes a `scikit-learn`-compatible
interface with ``KFold`` for cross-validation and ``train_test_split`` for
convenient train/test partitioning.

Features
--------
- **KFold**: Kennard–Stone based K-fold cross-validator (non-stratified).
- **train_test_split**: Kennard–Stone based train/test splitting utility.

Examples
--------
>>> from kennard_stone import KFold, train_test_split
>>> X_train, X_test = train_test_split(X, test_size=0.2)
>>> for train_idx, test_idx in KFold(n_splits=5).split(X):
...     pass

Notes
-----
Docstrings follow the NumPy/Google style (via napoleon) for Sphinx
autodocumentation.

Copyright © 2021 yu9824
"""

from ._core._core import KFold, train_test_split

__version__ = "3.0.1-rc.1"
__license__ = "MIT"
__author__ = "yu9824"
__copyright__ = "Copyright © 2021 yu9824"
__url__ = "https://github.com/yu9824/kennard_stone"

__all__ = (
    "KFold",
    "train_test_split",
)
