import scipy.spatial.distance

from kennard_stone import train_test_split

# Metrics removed from SciPy (and hence from scikit-learn's valid metrics).
# 'sokalmichener': deprecated in SciPy 1.15, removed in SciPy 1.17.
METRICS_REMOVED_FROM_SCIPY = ("sokalmichener",)


def test_train_test_split(prepare_data):
    X, y = prepare_data

    METRICS = (
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
    )

    for metric in METRICS:
        if metric in METRICS_REMOVED_FROM_SCIPY and not hasattr(
            scipy.spatial.distance, metric
        ):
            continue
        train_test_split(X, y, test_size=0.2, metric=metric, n_jobs=-1)
