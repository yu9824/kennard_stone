"""
This is an algorithm for evenly partitioning data in a `scikit-learn`-like
interface.

Copyright © 2021 yu9824
"""

from ._core._core import KFold, train_test_split

__version__ = "3.0.0rc1"
__license__ = "MIT"
__author__ = "yu9824"
__copyright__ = "Copyright © 2021 yu9824"
__url__ = "https://github.com/yu9824/kennard_stone"

__all__ = [
    "KFold",
    "train_test_split",
]
