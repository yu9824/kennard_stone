from sklearn.datasets import fetch_california_housing
import pytest

from kennard_stone import train_test_split
from kennard_stone.utils import is_installed

if is_installed("torch"):
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = None
else:
    device = None


@pytest.fixture
def prepare_data():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return (X, y)


@pytest.mark.skipif(device is None, reason="GPU is not available.")
def test_train_test_split_with_large_gpu(prepare_data):
    _ = train_test_split(*prepare_data, test_size=0.2, device=device)
