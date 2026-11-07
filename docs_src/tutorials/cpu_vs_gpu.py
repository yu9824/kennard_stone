# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CPU vs GPU

# %% [markdown]
# ## Device info
#
# The timings below were measured on:
#
# - MacBook Air, Apple M4 (10-core CPU: 4 performance and 6 efficiency cores), 16 GB memory
# - macOS 26.7.2
# - Python 3.10.20, PyTorch 2.11.0 (MPS), scikit-learn 1.7.2

# %%
from time import perf_counter

import numpy as np
from sklearn.datasets import fetch_california_housing

# You need to install 'pytorch' when you use GPU
import torch

from kennard_stone import train_test_split

# %%
X: np.ndarray
X, _ = fetch_california_housing(return_X_y=True)
X.shape

# %% [markdown]
# ## CPU

# %%
t_start = perf_counter()
_ = train_test_split(
    X, test_size=0.2, n_jobs=-1, device="cpu", metric="euclidean"
)
t_end = perf_counter()
t_cpu = t_end - t_start
print(f"CPU: {t_cpu:.3f} sec")

# %% [markdown]
# ## GPU

# %%
# cuda
if torch.cuda.is_available():
    device = "cuda"
# mps (Apple silicon)
elif torch.backends.mps.is_available():
    device = "mps"
else:
    raise RuntimeError("GPU is not available.")

t_start = perf_counter()
_ = train_test_split(
    X, test_size=0.2, n_jobs=-1, device=device, metric="euclidean"
)
t_end = perf_counter()
t_gpu = t_end - t_start
print(f"{device}: {t_gpu:.3f} sec")
