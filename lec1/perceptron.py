#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1回演習問題
"""
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from keras.datasets import mnist

use_small_data = False

##### データの取得 (MNIST 手書き数字データセット)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 学習データ
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape([60000, 28 * 28])
x_train = x_train[y_train <= 1, :]

# テストデータ
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape([10000, 28 * 28])
x_test = x_test[y_test <= 1, :]

y_train = np.array(y_train[y_train <= 1], float)
y_test = np.array(y_test[y_test <= 1], float)

if use_small_data:
    ## プログラム作成中は訓練データを小さくして，
    ## 実行時間が短くようにしておく
    y_train = y_train[range(1000)]
    x_train = x_train[range(1000), :]

n, d = x_train.shape
n_test, _ = x_test.shape

### 以下の３行のコメントを外すと訓練データの
### 一つ目の手書き数字の画像が'x_train0.pdf'に保存される
# plt.figure()
# plt.imshow(x_train[0,:].reshape([28,28]))
# plt.savefig('x_train0.pdf')

np.random.seed(123)


##### シグモイド関数, 誤差関数
def sigmoid(x):
    # 課題2(a)
    # returnの後にシグモイド関数を返すプログラムを書く
    return 1 / (1 + np.exp(-x))


def error_function(f, y):
    # 課題2(b)
    # returnの後に誤差関数を返すプログラムを書く
    return -y * np.log(f) - (1 - y) * np.log(1 - f)


##### パラメータの初期値 (ランダム)
w = np.random.normal(0, 0.3, d + 1)

########## 確率的勾配降下法によるパラメータ推定
error = []
error_test = []

num_epoch = 10

eta = 0.01

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)

    e_train = np.full(n, np.nan)  # 誤差保存用のサイズnの配列をNaNで初期化
    for i in index:

        xi = np.append(1, x_train[i, :])  # 特徴ベクトルに定数1を結合したものを作成
        yi = y_train[i]

        ### 課題2(a)
        # パーセプトロンの出力f(w^T x)を計算する (活性化関数fはシグモイド)
        # ヒント: 先に上でsigmoid関数を定義し,
        #         パラメータ変数'w'と入力'xi'の内積(np.dot)の結果を渡す
        # fi = シグモイドの出力 (i番目のx(xi)に対するfなのでfiとしてある)
        fi = sigmoid(np.dot(w, xi))

        ### 課題2(b) 誤差評価
        # fを計算したら，error_functionを定義し以下のコメントを外す
        e_train[i] = error_function(fi, yi)

        if epoch == 0:
            continue
            # 初期パラメータでの誤差を計算するためにepoch=0では何もしない
            # 注: continueはfor文の先頭に戻る命令
            # 注: 初期誤差を観察しないなら, 実際には最初から学習を開始してもよい

        ### 課題2(c)
        # パラメータの更新
        # 確率的勾配降下法の更新式を書く（学習率eta_tの設定に注意）
        w = w - eta * (fi - yi) * xi

    ##### エポックごとの訓練誤差
    error.append(sum(e_train) / n)

    ##### テスト誤差
    e_test = np.full(n_test, np.nan)
    for j in range(0, n_test):

        xi = np.append(1, x_test[j, :])
        yi = y_test[j]

        ### 課題2(a)
        # パーセプトロンの出力fを計算する(テスト)
        fi = sigmoid(np.dot(w, xi))

        ### 課題2(b) 誤差評価
        # fを計算したら，error_functionを定義し以下のコメントを外す
        e_test[j] = error_function(fi, yi)

    error_test.append(sum(e_test) / n_test)

########## 誤差関数のプロット
plt.figure()
plt.plot(error, label="training", lw=3)  # 青線
plt.plot(error_test, label="test", lw=3)  # オレンジ線
plt.yscale("log")
plt.grid()
plt.ylabel("Error function (log scale)")
plt.xlabel("Epoch")
plt.legend(fontsize=16)
plt.savefig("./error.pdf", bbox_inches="tight", transparent=True)
plt.close()
