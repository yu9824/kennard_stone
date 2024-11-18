from kennard_stone import train_test_split


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
        train_test_split(X, y, test_size=0.2, metric=metric, n_jobs=-1)
