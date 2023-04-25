from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
import pytest

from kennard_stone import train_test_split, KFold


@pytest.fixture
def prepare_data():
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target
    return (X, y)


def test_train_test_split(prepare_data):
    X, y = prepare_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    assert X_train.index[0] == y_train.index[0] == 224
    assert X_train.index[-1] == y_train.index[-1] == 123
    assert X_test.index[0] == y_test.index[0] == 68
    assert X_test.index[-1] == y_test.index[-1] == 203


def test_KFold(prepare_data):
    X, y = prepare_data
    estimator = RandomForestRegressor(random_state=334, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True)
    cross_validate(
        estimator,
        X,
        y,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=kf,
        return_train_score=True,
    )
