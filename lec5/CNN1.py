#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns

np.random.seed(123)


def ReLU(x):
    return x * (x > 0)


def scale(x):
    # 可視化のために平均+-3*標準偏差の中に値の範囲を制限
    z = x - np.mean(x)
    C = 3 * np.std(z)
    z = np.maximum(np.minimum(z, C), -C)
    return 255 * (z + C) / (2 * C)


def Convolution(X, V, act_func, H, M, padding, stride):
    # 課題1. Convolution関数の作成

    # W: 縦と横のサイズ
    # K: 入力チャネル数
    W, _, K = X.shape
    # スライド10ページを参考に畳み込み後のサイズを設定
    # ヒント: int()で実数値を囲むと切り捨てされる
    W_next = int((W + 2 * padding - H) / stride + 1)

    # paddingした後のX
    # ヒント: 0で初期化した以下の３次元配列の
    # 1次元目と2次元目にpaddingの内側の添字を指定してXを代入する
    # スライド９ページ目にはpadding=1の場合の図がある
    # これを参考に指定すべき添字を考えること
    # （縦と横，どこからどこまでに元のXを入れればよいか）
    X_padded = np.zeros((W + 2 * padding, W + 2 * padding, K))  # 全て0で初期化
    X_padded[padding : W + padding, padding : W + padding, :] = X
    # （参考: その他にnp.pad関数を使う方法もある）

    U = np.zeros((W_next, W_next, M))
    # M個のフィルタを適用するループ
    for m in range(M):
        #    Uの要素を計算するループ
        for i in range(W_next):
            for j in range(W_next):
                # U_ijmを計算する (スライド１３ページ)
                # ヒント: 畳み込みの対象となる入力の部分配列は以下で指定する (??に何が入るべきか考えよ):
                #         X_padded["i*ストライドサイズ":"i*?? + ??", "j"を使って1次元目と同様に指定, :]
                #         これと，今着目しているフィルター V[:,:,:,m] の要素ごとの積をとった後に和をとる
                #         (np.sumで配列要素全ての和がとれる)
                U[i, j, m] = np.sum(
                    X_padded[
                        i * stride : i * stride + H, j * stride : j * stride + H, :
                    ]
                    * V[:, :, :, m]
                )

    # Uの値に活性化関数を適用して返す
    return act_func(U)


def MaxPooling(Z_prev, H, padding, stride):
    # 課題2. MaxPooling関数の作成

    W, _, K = Z_prev.shape

    # MaxPooling後のサイズの設定
    # ヒント: 畳み込みのときと同じでよい
    W_next = int((W + 2 * padding - H) / stride + 1)

    # Padding
    # ヒント: 畳み込みでの処理を参考に作成せよ
    Z_prev_padded = np.zeros((W + 2 * padding, W + 2 * padding, K))  # 全て0で初期化
    Z_prev_padded[padding : W + padding, padding : W + padding, :] = Z_prev

    Z = np.zeros((W_next, W_next, K))
    # K個のチャネルについてループ (poolingでは各チャネルごとに最大値をとる)
    for k in range(K):
        #    Zの要素を計算するループ
        for i in range(W_next):
            for j in range(W_next):
                # 畳み込みのときを参考に，Z_prevでの対象となる要素を指定する
                # (1, 2次元目がmaxを探す対象となる部分配列で，3次元目はkを指定すればよい)
                Z[i, j, k] = np.max(
                    Z_prev_padded[
                        i * stride : i * stride + H, j * stride : j * stride + H, k
                    ]
                )

    return Z


plot_result = True  # 図をplotする場合はTrueに

# データの読み込み
X = np.load("./P64.npy")
# X = np.load("./L256.npy") # 注: こちらは画素が少し多いので計算が重い可能性有

# フィルタ
F_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
F_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
F_laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
F_smooth = np.array(
    [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
)
F_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
F_zero = np.zeros((3, 3))

# 画像のサイズ WxW
W = X.shape[0]

# WxWx1に変形
X = X.reshape(W, W, 1)

H = 3  # フィルタのサイズ
M = 3  # フィルタ数

# フィルタ (1回目)
# サイズは H x H x K x M
# Kはチャネル数（入力画像はK=1）
V0 = np.zeros((H, H, 1, M))

V0[:, :, 0, 0] = F_sobel_y
V0[:, :, 0, 1] = F_sobel_x
V0[:, :, 0, 2] = F_smooth

# 課題1 Convolutionの中身を作成せよ
Z1 = Convolution(X, V0, ReLU, H, M, 1, 1)

# 課題2 MaxPoolingの中身を作成せよ
Z2 = MaxPooling(Z1, 2, 1, 2)

# 課題3. 2回目のConv->MaxPoolingを作成

# フィルタ (2回目)
# ヒント: 上のV0の定義の仕方を参考にすること
#         ただし，ここでは入力チャネル数が３になる
#         (数式上の添字とpythonの添字のずれに注意)
V1 = np.zeros((H, H, 3, M))

###
# ここでV1の適当な部分配列に各F_***を設定
V1[:, :, 0, 0] = F_sharpen
V1[:, :, 1, 0] = F_sharpen
V1[:, :, 2, 0] = F_zero
V1[:, :, 0, 1] = F_smooth
V1[:, :, 1, 1] = F_smooth
V1[:, :, 2, 1] = F_zero
V1[:, :, 0, 2] = F_zero
V1[:, :, 1, 2] = F_zero
V1[:, :, 2, 2] = F_laplacian
###

Z3 = Z1 = Convolution(X, V1, ReLU, H, M, 1, 1)

Z4 = MaxPooling(Z3, 2, 1, 2)

if plot_result:
    # 結果の出力(編集不要)
    colmap = "gray"
    fig = plt.figure()
    ax = fig.add_subplot(3, 5, 6)
    ax.imshow(X[:, :, 0], cmap=colmap)
    ax.axis("off")
    plt.title("Original")

    plt_idx = 3
    index = np.array(range(1, 16)).reshape(3, 5).T.reshape(-1)
    for l in range(2):
        if l == 0:
            C = scale(Z1)
            P = scale(Z2)
        else:
            C = scale(Z3)
            P = scale(Z4)

        for i in range(C.shape[2]):
            ax = fig.add_subplot(3, 5, index[plt_idx])
            plt_idx += 1
            ax.imshow(C[:, :, i], cmap=colmap)
            ax.axis("off")
            plt.title("Z^(" + str(2 * l + 1) + ")_ij" + str(i + 1))

        for i in range(P.shape[2]):
            ax = fig.add_subplot(3, 5, index[plt_idx])
            plt_idx += 1
            ax.imshow(P[:, :, i], cmap=colmap)
            ax.axis("off")
            plt.title("Z^(" + str(2 * (l + 1)) + ")_ij" + str(i + 1))

    plt.savefig("./cnn.pdf", bbox_inches="tight", transparent=True)

    plt.close()
