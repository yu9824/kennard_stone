import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from kennard_stone.utils import is_installed, pairwise_distances

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
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return (X, y)


@pytest.mark.skipif(device is None, reason="GPU is not available.")
def test_gpu_euclidian(prepare_data):
    # 通常の距離行列
    X, _ = prepare_data
    X: np.ndarray

    distance_X = pairwise_distances(X, metric="euclidean", device="cpu")
    distance_X_gpu = pairwise_distances(X, metric="euclidean", device=device)

    assert np.allclose(distance_X, distance_X_gpu)

    # 平均との行列

    distance_X_mean = pairwise_distances(
        X, X.mean(axis=0, keepdims=True), metric="euclidean", device="cpu"
    )
    distance_X_mean_gpu = pairwise_distances(
        X, X.mean(axis=0, keepdims=True), metric="euclidean", device=device
    )
    assert np.allclose(distance_X_mean, distance_X_mean_gpu)


@pytest.mark.skipif(device is None, reason="GPU is not available.")
def test_gpu_manhattan(prepare_data):
    # 通常の距離行列
    X, _ = prepare_data
    X: np.ndarray

    distance_X = pairwise_distances(X, metric="manhattan")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Convert NumPy array to PyTorch tensor and move it to GPU
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Calculate pairwise distances on GPU
    distance_X_gpu = torch.cdist(X_tensor, X_tensor, p=1)

    # Move the result back to CPU if needed
    distance_X_gpu = distance_X_gpu.fill_diagonal_(0).cpu().numpy()

    assert np.allclose(distance_X, distance_X_gpu)

    # 平均との行列

    distance_X_mean = pairwise_distances(
        X, X.mean(axis=0, keepdims=True), metric="manhattan"
    )

    distance_X_mean_gpu = torch.cdist(
        X_tensor, X_tensor.mean(axis=0, keepdims=True), p=1
    )

    assert np.allclose(distance_X_mean, distance_X_mean_gpu.cpu().numpy())


@pytest.mark.skipif(device is None, reason="GPU is not available.")
def test_gpu_chebyshev(prepare_data):
    # 通常の距離行列
    X, _ = prepare_data
    X: np.ndarray

    distance_X = pairwise_distances(X, metric="chebyshev")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Convert NumPy array to PyTorch tensor and move it to GPU
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Calculate pairwise distances on GPU
    distance_X_gpu = torch.cdist(X_tensor, X_tensor, p=float("inf"))

    # Move the result back to CPU if needed
    distance_X_gpu = distance_X_gpu.fill_diagonal_(0).cpu().numpy()

    assert np.allclose(distance_X, distance_X_gpu)

    # 平均との行列

    distance_X_mean = pairwise_distances(
        X, X.mean(axis=0, keepdims=True), metric="chebyshev"
    )

    distance_X_mean_gpu = torch.cdist(
        X_tensor, X_tensor.mean(axis=0, keepdims=True), p=float("inf")
    )

    assert np.allclose(distance_X_mean, distance_X_mean_gpu.cpu().numpy())
