from __future__ import annotations

from typing import Literal

import pytest

from kennard_stone.utils import is_installed

if is_installed("torch"):
    import torch  # type: ignore


@pytest.fixture(scope="session")
def get_device() -> Literal["cuda", "mps", "cpu"]:
    device: Literal["cuda", "mps", "cpu"]
    if is_installed("torch"):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"
    return device
