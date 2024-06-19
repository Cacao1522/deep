#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2回演習問題1
"""
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical

# from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

use_small_data = False  # True
plot_misslabeled = True # False  

# クラス数
m = 4

##### データの取得
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape([60000, 28 * 28])
x_train = x_train[y_train < m, :]

x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape([10000, 28 * 28])
x_test = x_test[y_test < m, :]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

if use_small_data:
    ## プログラム作成中は訓練データを小さくして，
    ## 実行時間が短くようにしておく
    y_train = y_train[range(1000)]
    x_train = x_train[range(1000), :]

n, d = x_train.shape
n_test, _ = x_test.shape

np.random.seed(123)


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    ## 課題1(a)
    ## ReLUの微分を返すプログラムを書く
    return x > 0


def softmax(x):
    ##### 課題1(c)
    # ソフトマックス関数を返すプログラムを書く

    return np.exp(x) / np.sum(np.exp(x))


def CrossEntoropy(g, y):
    ##### 課題1(d)
    # クロスエントロピーを返すプログラムを書く

    return -np.dot(y, np.log(g))


def forward(x, W, actfunc, d_actfunc):
    ##### 課題1(b)
    # 順伝播のプログラムを書く
    # actfuncとd_actfuncは関数型の引数であり,
    # ReLUとその微分を返す「関数」であることに注意

    # W*xを計算
    W_x = np.dot(W, x)

    # zを作る (定数"1"とactfuncの出力をappend)
    z = np.append(1, actfunc(W_x))

    # nabla f(W*x)を計算
    nabla_f = d_actfunc(W_x)

    return z, nabla_f  # 二つ返り値を返す


def backward(V_tilde, delta, derivative):
    ##### 課題1(e)
    # 各引数の表現するものは，
    # すでに記載済みの呼び出し元を参照して確認すること

    # 逆伝播のプログラムを書く

    return np.dot(V_tilde.T, delta) * derivative


#####中間層のユニット数とパラメータの初期値
q = 200
W = np.random.normal(0, 0.3, size=(q, d + 1))
V = np.random.normal(0, 0.3, size=(m, q + 1))

########## 確率的勾配降下法によるパラメータ推定
error = []
error_test = []

num_epoch = 10

eta = 0.1

for epoch in range(0, num_epoch):
    print("epoch =", epoch)
    index = np.random.permutation(n)

    e = np.full(n, np.nan)  # 誤差保存用のサイズnの配列をNaNで初期化
    for i in index:
        xi = np.append(1, x_train[i, :])
        yi = y_train[i, :]

        ##### 順伝播

        # 課題1(b)でforwardの中を作成
        # (最後二つの引数は関数型の引数であり，
        # 1(a)でdReLUが定義してある必要がある)

        z, nabla_f = forward(xi, W, ReLU, dReLU)

        # 課題1(c)でsoftmaxの中身を定義すると,
        # ネットワークの出力がgに入る (m次元ベクトル)
        g = softmax(np.dot(V, z))

        ##### 誤差評価

        # 課題1(d)でCrossEntropy関数を定義する
        e[i] = CrossEntoropy(g, yi)

        if epoch == 0:
            continue

        eta_t = eta / epoch

        ##### 逆伝播
        # 課題1(e)でdelta2とdelta1を作成
        # delta1はbackward関数の中身を作成

        delta2 = g - yi

        # 逆伝播計算ではVの最初の列は省く
        # （定数ユニットは逆伝播しない）
        delta1 = backward(V[:, 1:], delta2, nabla_f)

        ##### パラメータの更新
        # 課題1(f)で更新式を作成
        V = V - eta_t * np.outer(delta2, z.T)
        W = W - eta_t * np.outer(delta1, xi.T)

    ##### training error
    error.append(sum(e) / n)

    ##### test error
    e_test = np.full(n_test, np.nan)
    for j in range(0, n_test):
        xi = np.append(1, x_test[j, :])
        yi = y_test[j, :]

        z, nabla_f = forward(xi, W, ReLU, dReLU)
        g = softmax(np.dot(V, z))

        e_test[j] = CrossEntoropy(g, yi)

    error_test.append(sum(e_test) / n_test)
    e_test = []

########## 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)  # 青線
plt.plot(error_test, label="test", lw=3)  # オレンジ線
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy (log-scale)")
plt.grid()
plt.legend(fontsize=16)
plt.savefig("./error.pdf", transparent=True)

########## 確率が高いクラスにデータを分類
##### モデルの出力を評価
prob = []
for j in range(0, n_test):
    xi = np.append(1, x_test[j, :])
    yi = y_test[j, :]

    z, nabla_f = forward(xi, W, ReLU, dReLU)
    g = softmax(np.dot(V, z))

    prob.append(g)

predict = np.argmax(prob, axis=1)

predict_label = np.argmax(prob, axis=1)
true_label = np.argmax(y_test, axis=1)

##### 課題2
# confusion matrixを完成させる
ConfMat = np.zeros((m, m))
for i in range(m):
    for j in range(m):  
        ConfMat[i,j]=np.sum((true_label==i)&(predict_label==j)) 

# ここでConfMatの各要素を計算

plt.clf()
fig, ax = plt.subplots(figsize=(5, 5))
# fig.show()
sns.heatmap(
    ConfMat.astype(dtype=int),
    linewidths=1,
    annot=True,
    fmt="1",
    cbar=False,
    cmap="Blues",
)
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)

print("Plotting confusion matrices ...")

# 誤分類結果のプロット
if plot_misslabeled:
    n_maxplot = 20
    n_plot = 0
    for i in range(m):
        idx_true = y_test[:, i] == 1
        for j in range(m):
            if j != i:
                idx_predict = predict == j
                for l in np.where(idx_true * idx_predict == True)[0]:
                    plt.clf()
                    D = np.reshape(x_test[l, :], (28, 28))
                    sns.heatmap(D, cbar=False, cmap="Blues", square=True)
                    plt.axis("off")
                    plt.title("{} to {}".format(i, j))
                    plt.savefig(
                        "./misslabeled{}.pdf".format(l),
                        bbox_inches="tight",
                        transparent=True,
                    )
                    n_plot += 1
                    if n_plot >= n_maxplot:
                        break
            if n_plot >= n_maxplot:
                break
        if n_plot >= n_maxplot:
            break

    plt.close()
