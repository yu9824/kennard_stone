# Kennard Stone
## 概要
均等に分割するためのアルゴリズム．（詳しくは[参考文献](#参考文献)参照）<br>
train_test_splitやKFold，cross_val_scoreを用意．

## 使い方
### kennard_stoneの場合
```python
from kennard_stone import KennardStone
ks = KennardStone()
X_train, X_test, y_train, y_test = ks.train_test_split(X, y, test_size = 0.2)
```
その他の使い方はexampleフォルダを参照．

### scikit-learnの場合（参考）
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 334)
```


## 注意点
分割がデータセットに対して一通りに決まるので，```random_state```や```shuffle```という概念がない．<br>
それらを引数に入れてしまうとerrorを生じうる．<br><br>
また，version: 0.0.3時点では```n_jobs```も未実施．


## 参考文献
### 論文
http://www.tandfonline.com/doi/abs/10.1080/00401706.1969.10490666
### サイト
https://datachemeng.com/trainingtestdivision/