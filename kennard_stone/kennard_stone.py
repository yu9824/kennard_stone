import pandas as pd
import numpy as np
from math import floor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer, get_scorer
from sklearn.base import clone  # 新しいけどパラメータが一緒のestimatorを作成
from ngboost import NGBRegressor

'''
x : 特徴量
select_size : train_size (0以上1未満)

selected_sample_numbers : selected sample numbers (training data)
remaining_sample_numbers : remaining sample numbers (test data)
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

        '''
        # make_scorerのobjectをうけとるのでこの操作は不要．したのメソッドで置換される．
        estimator.fit(X_train, y_train)
        y_pred_on_test = estimator.predict(X_test)
        scores.append(scoring(y_test, y_pred_on_test))
        '''
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


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    rf = RandomForestRegressor(random_state = 334, n_estimators = 100)

    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 334)

    # from pdb import set_trace
    # print(rf)
    # scores = cross_val_score_KS(rf, X, y, cv = 5, scoring = mean_absolute_error)
    rf.fit(X_train, y_train)
    print(get_scorer('neg_mean_squared_error')(rf, X_test, y_test))
    print(get_scorer(make_scorer(mean_squared_error))(rf, X_test, y_test))
    print(get_scorer('neg_mean_squared_error'))
    f = mean_squared_error
    print(str(mean_squared_error))

    # print(scores)
    # print(cross_val_score_KS(rf, df.iloc[:, 1:], df.iloc[:, 0], cv = 5, scoring = mean_absolute_error))



    # a = np.expand_dims(np.arange(24).reshape(4, 6), 0)
    # b = np.expand_dims(np.ones(30).reshape(5, 6), 1)

    # print(a - b)
