if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.model_selection import cross_validate

    from kennard_stone import kennard_stone
    from kennard_stone import _deprecated

    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target

    ks_old = _deprecated._KennardStone()
    ks_new = kennard_stone._KennardStone(n_groups=1)

    assert ks_old._get_indexes(X) == ks_new.get_indexes(X)[0]
