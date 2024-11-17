from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing, load_diabetes


@pytest.fixture(scope="session")
def prepare_data() -> tuple[np.ndarray, np.ndarray]:
    return load_diabetes(return_X_y=True)


@pytest.fixture(scope="session")
def prepare_large_data() -> tuple[np.ndarray, np.ndarray]:
    return fetch_california_housing(return_X_y=True)
