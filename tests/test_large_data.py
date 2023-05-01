# from sklearn.datasets import fetch_california_housing
# import pytest

# from kennard_stone import train_test_split


# @pytest.fixture
# def prepare_data():
#     data = fetch_california_housing(as_frame=True)
#     X = data.data
#     y = data.target
#     return (X, y)


# def test_train_test_split_with_large(prepare_data):
#     X_train, X_test, y_train, y_test = train_test_split(
#         *prepare_data, test_size=0.2, n_jobs=-1
#     )
