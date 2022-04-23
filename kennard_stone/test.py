from kennard_stone import train_test_split, KFold
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
import pandas as pd

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name = 'PRICE')

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
print(X_train, y_train, X_test, y_test)

estimator = RandomForestRegressor(random_state=334, n_jobs=-1)

for kf in (KFold(n_splits=5, alternate=True), KFold(n_splits=5, alternate=False)):
    print(cross_validate(estimator, X, y, scoring='neg_mean_squared_error', n_jobs = -1, cv = kf, return_train_score=True))
