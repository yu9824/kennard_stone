import sys
from typing import Literal, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from ._utils import is_installed

if is_installed("torch"):
    import torch  # type: ignore

    Device: TypeAlias = Union[
        "torch.device", str, Literal["cpu", "cuda", "mps"]
    ]
else:
    Device: TypeAlias = Union[str, Literal["cpu"]]  # type: ignore[misc,no-redef]

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
