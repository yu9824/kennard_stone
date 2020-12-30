import pandas as pd
import numpy as np
from math import floor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer, get_scorer
from sklearn.base import clone  # 新しいけどパラメータが一緒のestimatorを作成

'''
x : 特徴量
select_size : train_size (0以上1未満)

selected_sample_numbers : selected sample numbers (training data)
remaining_sample_numbers : remaining sample numbers (test data)
'''

class KennardStone:
    def __init__(self):
        pass

    def __call__(self, X):
        # np.ndarray化
        X = np.array(X)

        # もとのXを取っておく
        self.original_X = X.copy()

        # 全ての組成に対してそれぞれの平均との距離の二乗を配列として得る． (サンプル数の分だけ存在)
        self.distance_to_ave = np.sum((X - X.mean(axis = 0)) ** 2, axis = 1)

        # 最大値を取るサンプル　(平均からの距離が一番遠い) のindex_numberを保存
        i_farthest = np.argmax(self.distance_to_ave)

        # 抜き出した (train用) サンプルのindex_numberを保存しとくリスト
        i_selected = [i_farthest]

        # まだ抜き出しておらず，残っているサンプル (test用) サンプルのindex_numberを保存しておくリスト
        i_remaining = np.arange(len(X))

        # 抜き出した (train用) サンプルに選ばれたサンプルをtrain用のものから削除
        X = np.delete(X, i_selected, axis = 0)
        i_remaining = np.delete(i_remaining, i_selected, axis = 0)

        return self._sort(X, i_selected, i_remaining)

    def _sort(self, X, i_selected, i_remaining):
        # 選ばれたサンプル (x由来)
        samples_selected = self.original_X[i_selected, :]

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







'''
def cross_val_score_KS(estimator, X, y, **options):
    cv = int(options['cv']) if 'cv' in options else 5
    scoring = get_scorer(options['scoring']) if 'scoring' in options else mean_squared_error

    # 念のため， Xとyをnp.ndarrayに変換
    X = np.array(X)
    y = np.array(y)

    # 分割したものをそれぞれいれとくやつ．
    kfold = {'data':[], 'index':[]}
    indexes = np.arange(len(X))
    def split(X, y, indexes, kfold):
        X1, X2, y1, y2, i1, i2 = train_test_split_KS(X, y, indexes, test_size = 1 / (cv - len(kfold['index'])))
        kfold['data'].append([X2, y2])
        kfold['index'].append(i2)
        if len(kfold['index']) == cv:
            return kfold
        else:
            return split(X1, y1, i1, kfold)

    # k分割
    kfold = split(X, y, indexes, kfold)

    # output用． スコアをまとめとくやつ．
    # 辞書にして複数出せるようにしてもいいけど，，，
    scores = []
    for iteration in range(cv):    # どのデータをtestデータにするか．
        X_test, y_test = kfold['data'][iteration]
        boolean = [False if i in kfold['index'][iteration] else True for i in range(len(X))]
        X_train = X[boolean, :]
        y_train = y[boolean]

        # # make_scorerのobjectをうけとるのでこの操作は不要．したのメソッドで置換される．
        # estimator.fit(X_train, y_train)
        # y_pred_on_test = estimator.predict(X_test)
        # scores.append(scoring(y_test, y_pred_on_test))

        estimator = clone(estimator)
        if isinstance(estimator, NGBRegressor):
            X_train1, X_train2, y_train1, y_train2 = train_test_split_KS(X_train, y_train, test_size = 0.1)
            estimator.fit(X_train1, y_train1, X_val = X_train2, Y_val = y_train2)
        else:
            estimator.fit(X_train, y_train)
        scores.append(scoring(estimator, X_test, y_test))
    return scores


def train_test_split_KS(X, *arrays, **options):
    if 'train_size' in options:
        select_size = options['train_size']
    elif 'test_size' in options:
        select_size = 1 - options['test_size']
    else:
        select_size = 0.75

    train_indexes, test_indexes = KennardStone(X, select_size = select_size)
    lst_output = list(extract(X, train_indexes, test_indexes))

    if len(arrays) != 0:
        for ar in arrays:
            if len(ar) == len(X):
                lst_output.extend(extract(ar, train_indexes, test_indexes))
            else:
                exit('サイズが違う配列が含まれているので無理です．')
    return lst_output



def KennardStone(x, select_size = 0.7):
    # np.ndarray化
    x = np.array(x)

    # もとのxを取っておく．
    original_x = x.copy()

    # 全ての組成に対してそれぞれの平均との距離の二乗を配列として得る． (サンプル数の分だけ存在)
    distance_to_average = np.sum((x - x.mean(axis = 0)) ** 2, axis = 1)

    # 最大値を取るサンプル　(平均からの距離が一番遠い) のindex_numberを保存
    max_distance_sample_number = np.argmax(distance_to_average)

    # 抜き出した (train用) サンプルのindex_number
    selected_sample_numbers = [max_distance_sample_number]

    # 残っているサンプル (test用) サンプルのindex_number
    remaining_sample_numbers = np.arange(len(x))

    # 抜き出した (train用) サンプルに選ばれたサンプルをtrain用のものから削除
    x = np.delete(x, selected_sample_numbers, axis = 0)
    remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, axis = 0)

    k = floor(len(x) * select_size)

    def select(x, selected_sample_numbers, remaining_sample_numbers):
        # 選ばれたサンプル (x由来)
        selected_samples = original_x[selected_sample_numbers, :]

        # まだ選択されていない各サンプルにおいて、これまで選択されたすべてのサンプルとの間でユークリッド距離を計算する
        min_distance_to_selected_samples = np.min(np.sum((np.expand_dims(selected_samples, 1) - np.expand_dims(x, 0)) ** 2, axis = 2), axis = 0)

        # 最大値を取るサンプル　(距離が一番遠い) のindex_numberを保存
        max_distance_sample_number = np.argmax(min_distance_to_selected_samples)

        selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
        x = np.delete(x, max_distance_sample_number, axis = 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)

        if len(selected_sample_numbers) < k:
            return select(x, selected_sample_numbers, remaining_sample_numbers)
        else:
            return selected_sample_numbers, remaining_sample_numbers

    return select(x, selected_sample_numbers, remaining_sample_numbers)

def extract(array, train_indexes, test_indexes):
    if isinstance(array, pd.DataFrame):
        return array.iloc[train_indexes], array.iloc[test_indexes]
    else:
        return array[train_indexes], array[test_indexes]
'''

if __name__ == '__main__':
    pass



