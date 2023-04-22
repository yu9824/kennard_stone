if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_validate

    from kennard_stone import train_test_split, KFold

    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train, y_train, X_test, y_test)

    estimator = RandomForestRegressor(random_state=334, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True)
    print(
        cross_validate(
            estimator,
            X,
            y,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            cv=kf,
            return_train_score=True,
        )
    )
