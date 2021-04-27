from kennard_stone import train_test_split
from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name = 'PRICE')

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
print(X_train, y_train, X_test, y_test)