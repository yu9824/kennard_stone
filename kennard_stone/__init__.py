"""
This is an algorithm for evenly partitioning data in a `scikit-learn`-like
interface.

Copyright © 2021 yu9824
"""

from .kennard_stone import KFold, train_test_split

__version__ = "2.2.1"
__license__ = "MIT"
__author__ = "yu9824"
__copyright__ = "Copyright © 2021 yu9824"
__url__ = "https://github.com/yu9824/kennard_stone"

__all__ = [
    "KFold",
    "train_test_split",
]
