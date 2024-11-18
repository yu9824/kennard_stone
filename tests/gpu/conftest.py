from __future__ import annotations

from typing import Literal

import pytest

from kennard_stone.utils import is_installed


@pytest.fixture(scope="session")
def get_device() -> Literal["cuda", "mps", "cpu"]:
    device: Literal["cuda", "mps", "cpu"]
    if is_installed("torch"):
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"
    print(device)
    return device
