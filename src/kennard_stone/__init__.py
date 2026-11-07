"""Split data evenly with the Kennard–Stone algorithm.

The package provides a scikit-learn compatible interface: :class:`KFold` for
cross-validation and :func:`train_test_split` for a train/test split.

Copyright © 2021 yu9824
"""

from importlib.metadata import PackageNotFoundError, version

from ._core._core import KFold, train_test_split

try:
    __version__ = version("kennard-stone")
except PackageNotFoundError:
    # Not installed, e.g. imported from the source tree via PYTHONPATH.
    __version__ = "unknown"
__license__ = "MIT"
__author__ = "yu9824"
__copyright__ = "Copyright © 2021 yu9824"
__url__ = "https://github.com/yu9824/kennard_stone"

__all__ = (
    "KFold",
    "train_test_split",
)
