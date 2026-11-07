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
# # Other distance metrics
#
# Compare the training and test splits obtained with different distance metrics (`metric`).

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

from kennard_stone import train_test_split

# %%
X: pd.DataFrame
y: pd.Series
X, y = load_iris(as_frame=True, return_X_y=True)
display(X.head(), y.to_frame().head())

# %%
X_train_euc, X_test_euc, y_train_euc, y_test_euc = train_test_split(
    X, y, test_size=0.2, n_jobs=-1, metric="euclidean"
)

# %%
X_train_mah, X_test_mah, y_train_mah, y_test_mah = train_test_split(
    X, y, test_size=0.2, n_jobs=-1, metric="mahalanobis"
)

# %%
fig, ax = plt.subplots(dpi=144)
sns.heatmap(
    X.corr(),
    vmax=1,
    vmin=-1,
    cbar_kws=dict(label="corr"),
    cmap="bwr",
    annot=True,
    fmt=".2f",
    ax=ax,
)

# %%
sns.pairplot(
    pd.concat(
        (
            pd.concat(
                (
                    X_train_euc,
                    pd.Series(
                        ("train",) * X_train_euc.shape[0],
                        name="label",
                        index=X_train_euc.index,
                    ),
                ),
                axis=1,
            ),
            pd.concat(
                (
                    X_test_euc,
                    pd.Series(
                        ("test",) * X_test_euc.shape[0],
                        name="label",
                        index=X_test_euc.index,
                    ),
                ),
                axis=1,
            ),
        ),
        axis=0,
    ),
    hue="label",
)

# %%
sns.pairplot(
    pd.concat(
        (
            pd.concat(
                (
                    X_train_mah,
                    pd.Series(
                        ("train",) * X_train_mah.shape[0],
                        name="label",
                        index=X_train_mah.index,
                    ),
                ),
                axis=1,
            ),
            pd.concat(
                (
                    X_test_mah,
                    pd.Series(
                        ("test",) * X_test_mah.shape[0],
                        name="label",
                        index=X_test_mah.index,
                    ),
                ),
                axis=1,
            ),
        ),
        axis=0,
    ),
    hue="label",
)
