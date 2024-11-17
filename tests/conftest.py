from __future__ import annotations

from logging import DEBUG

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing, load_diabetes

from kennard_stone.logging import get_root_logger


@pytest.fixture(scope="session")
def prepare_data() -> tuple[np.ndarray, np.ndarray]:
    return load_diabetes(return_X_y=True, as_frame=True)


@pytest.fixture(scope="session")
def prepare_large_data() -> tuple[np.ndarray, np.ndarray]:
    return fetch_california_housing(return_X_y=True, as_frame=True)


@pytest.fixture(scope="session", autouse=True)
def setup_logger():
    get_root_logger().setLevel(DEBUG)
