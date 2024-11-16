from typing import Literal, TypeAlias, Union

from .utils import is_installed

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
