"""
This program was used in v1. It is implemented in a foolproof manner and is
left for checking the calculation results.

We do not recommend users to use it because it is very slow with no parallel
computation implemented, and the problem is that it only implements
the KennardStone method using Euclidean distance.

Copyright © 2021 yu9824
"""


import numpy as np

from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler


# from kennard_stone import _KennardStoneでは呼び出せない．
# したがって，呼び出したい場合は，import kennard_stone
# _KennardStone = kennard_stone.kennard_stone._KennardStoneとする必要がある．
class _KennardStone:
    def __init__(self, scale=True):
        """The root program of the Kennard-Stone algorithm.

        Parameters
        ----------
        scale : bool, optional
            scaling X or not, by default True
        """
        self.scale = scale

    def _get_indexes(self, X):
        # check input array
        X: np.ndarray = check_array(X, ensure_2d=True)

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Save the original X.
        self._original_X = X.copy()

        # 全ての組成に対してそれぞれの平均との距離の二乗を配列として得る． (サンプル数の分だけ存在)
        distance_to_ave = np.sum(np.square(X - X.mean(axis=0)), axis=1)

        # 最大値を取るサンプル (平均からの距離が一番遠い) のindex_numberを保存
        i_farthest = np.argmax(distance_to_ave)

        # 抜き出した (train用) サンプルのindex_numberを保存しとくリスト
        i_selected = [i_farthest]

        # まだ抜き出しておらず，残っているサンプル (test用) サンプルのindex_numberを保存しておくリスト
        i_remaining = np.arange(len(X))

        # 抜き出した (train用) サンプルに選ばれたサンプルをtrain用のものから削除
        X = np.delete(X, i_selected, axis=0)
        i_remaining = np.delete(i_remaining, i_selected, axis=0)

        # 遠い順のindexのリスト．i.e. 最初がtrain向き，最後がtest向き
        indexes = self._sort(X, i_selected, i_remaining)

        return list(reversed(indexes))

    def _sort(self, X, i_selected, i_remaining):
        # 選ばれたサンプル (x由来)
        samples_selected = self._original_X[i_selected]

        # まだ選択されていない各サンプルにおいて、これまで選択されたすべてのサンプルとの間で
        # ユークリッド距離を計算し，その最小の値を「代表長さ」とする．
        min_distance_to_samples_selected = np.min(
            np.sum(
                (np.expand_dims(samples_selected, 1) - np.expand_dims(X, 0))
                ** 2,
                axis=2,
            ),
            axis=0,
        )

        # 最大値を取るサンプル　(距離が一番遠い) のindex_numberを保存
        i_farthest = np.argmax(min_distance_to_samples_selected)

        # 選んだとして記録する
        i_selected.append(i_remaining[i_farthest])
        X = np.delete(X, i_farthest, axis=0)
        i_remaining = np.delete(i_remaining, i_farthest, 0)

        if len(i_remaining):  # まだ残っているなら再帰
            return self._sort(X, i_selected, i_remaining)
        else:  # もうないなら終える
            return i_selected
