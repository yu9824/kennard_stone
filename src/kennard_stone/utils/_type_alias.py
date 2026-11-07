"""Type aliases used across the package.

This module defines ``Device`` for GPU/CPU device specification,
``Metrics`` as a literal set of metric names that can be used for distance
computation, and ``MetricLike`` accepting either a metric name or a callable.

Notes
-----
Representative public types referenced by the API are centralized here for
clear Sphinx rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from ._utils import is_installed

if is_installed("torch"):
    import torch  # type: ignore

    Device: TypeAlias = torch.device | str | Literal["cpu", "cuda", "mps"]
else:
    Device: TypeAlias = str | Literal["cpu"]  # type: ignore[misc,no-redef]

Metrics: TypeAlias = Literal[
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]

MetricLike: TypeAlias = Metrics | Callable[[ArrayLike, ArrayLike], np.ndarray]
"""Metric name or a callable returning a distance matrix."""

InitMethod: TypeAlias = Literal["farthest_from_mean", "farthest_pair"]
"""How to choose the first selected samples in the Kennard–Stone algorithm."""
