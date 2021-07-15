'''
Copyright © 2021 yu9824
'''

import pandas as pd
import numpy as np
from math import ceil

from itertools import chain
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils import indexable, _safe_indexing
from sklearn.preprocessing import StandardScaler

class KFold(_BaseKFold):
    def __init__(self, n_splits = 5, **kwargs):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        del self.shuffle
        del self.random_state
    
    def _iter_test_indices(self, X=None, y=None, groups=None):
        n_samples = _num_samples(X)

        _ks = _KennardStone()
        indices = _ks._get_indexes(X)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype = int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

class KSSplit(BaseShuffleSplit):
    @_deprecate_positional_args
    def __init__(self, n_splits=10, *, test_size=None, train_size=None):
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size)
        self._default_test_size = 0.1

    def _iter_indices(self, X, y=None, groups=None):
        _ks = _KennardStone()
        inds = _ks._get_indexes(X)

        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size, default_test_size = self._default_test_size)

        for _ in range(self.n_splits):
            ind_test = inds[:n_test]
            ind_train = inds[n_test:(n_test + n_train)]
            yield ind_train, ind_test

def train_test_split(*arrays, test_size=None, train_size=None, **kwargs):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size, default_test_size = 0.25)

    CVClass = KSSplit
    cv = CVClass(test_size=n_test, train_size=n_train)

    train, test = next(cv.split(X = arrays[0]))

    return list(chain.from_iterable((_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays))


# from kennard_stone import _KennardStoneでは呼び出せない．
# したがって，呼び出したい場合は，import kennard_stone; _KennardStone = kennard_stone.kennard_stone._KennardStoneとする必要がある．
class _KennardStone:
    # 引数には入れているが，基本的にFalseにすることはない．
    def __init__(self, scale = True, prior = 'test'):
        self.scale = scale
        self.prior = prior
    
    def _get_indexes(self, X):
        # np.ndarray化
        X = np.array(X)

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # もとのXを取っておく
        self._original_X = X.copy()

        # 全ての組成に対してそれぞれの平均との距離の二乗を配列として得る． (サンプル数の分だけ存在)
        distance_to_ave = np.sum((X - X.mean(axis = 0)) ** 2, axis = 1)

        # 最大値を取るサンプル (平均からの距離が一番遠い) のindex_numberを保存
        i_farthest = np.argmax(distance_to_ave)

        # 抜き出した (train用) サンプルのindex_numberを保存しとくリスト
        i_selected = [i_farthest]

        # まだ抜き出しておらず，残っているサンプル (test用) サンプルのindex_numberを保存しておくリスト
        i_remaining = np.arange(len(X))

        # 抜き出した (train用) サンプルに選ばれたサンプルをtrain用のものから削除
        X = np.delete(X, i_selected, axis = 0)
        i_remaining = np.delete(i_remaining, i_selected, axis = 0)

        # 遠い順のindexのリスト．i.e. 最初がtrain向き，最後がtest向き
        indexes = self._sort(X, i_selected, i_remaining)

        if self.prior == 'test':
            return list(reversed(indexes))
        elif self.prior == 'train':
            return indexes
        else:
            raise NotImplementedError

    def _sort(self, X, i_selected, i_remaining):
        # 選ばれたサンプル (x由来)
        samples_selected = self._original_X[i_selected]

        # まだ選択されていない各サンプルにおいて、これまで選択されたすべてのサンプルとの間でユークリッド距離を計算し，その最小の値を「代表長さ」とする．
        min_distance_to_samples_selected = np.min(np.sum((np.expand_dims(samples_selected, 1) - np.expand_dims(X, 0)) ** 2, axis = 2), axis = 0)

        # 最大値を取るサンプル　(距離が一番遠い) のindex_numberを保存
        i_farthest = np.argmax(min_distance_to_samples_selected)

        # 選んだとして記録する
        i_selected.append(i_remaining[i_farthest])
        X = np.delete(X, i_farthest, axis = 0)
        i_remaining = np.delete(i_remaining, i_farthest, 0)

        if len(i_remaining):   # まだ残っているなら再帰
            return self._sort(X, i_selected, i_remaining)
        else:   # もうないなら終える
            return i_selected


if __name__ == '__main__':
    from pdb import set_trace
    from sklearn.model_selection import cross_validate
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error as mse

    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    rf = RandomForestRegressor(n_jobs=-1, random_state=334)
    rf.fit(X_train, y_train)
    print(mse(rf.predict(X_test), y_test))

    kf = KFold(n_splits=5)
    print(cross_validate(rf, X, y, scoring = 'neg_mean_squared_error', cv = kf))



