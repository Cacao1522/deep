#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3回演習問題
"""
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist

use_small_data = True # False
index_plot_img = 0 # 元画像と復元画像するtest画像のindex
                   # 0からn_test-1までのいずれかの値を入れておく

##### データの取得
#クラス数
m = 1

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape([60000, 28*28])
x_train = x_train[y_train < m,:]

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape([10000, 28*28])
x_test = x_test[y_test < m,:]

if use_small_data:
    ## プログラム作成中は訓練データを小さくして，
    ## 実行時間が短くようにしておく
    x_train = x_train[range(1000),:]

n, d = x_train.shape
n_test, _ = x_test.shape

np.random.seed(123)

##### 
def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp, tmp*(1-tmp)

def ErrorFunction(x, y):
    ### 課題1
    # returnの後に二乗誤差関数を返すプログラムを書く
    return np.sum((y - x)**2)/2

def forward(z_in, W_k, actfunc): 
    ### forward関数は前回と同じでよい (DNN.py)
    ### W_kとz_inの内積を計算
    W_x = np.dot(W_k, z_in)
    ### actfuncから出力される活性化関数の値と微分値を保存
    f, nabla_f = actfunc(W_x)
    #
    ### 1と活性化関数の値fをappendしたものをz_outに設定
    z_out = np.append(1, f)
    #
    ### z_out, nabla_fを返す
    return z_out, nabla_f

def backward(W_tilde, delta, derivative): 
    ### backword関数も前回と同じでよい (DNN.py)
    return np.dot(W_tilde.T, delta) * derivative

### 課題4 adam関数
# 注1: 引数のW, m, vが数式中W^(t-1), m_t-1,, v_t-1に対応
# 注2: 値の書かれた引数はデフォルト引数であり，
# 呼び出し時に省略すると指定された値が入る
# 注3: tは関数の外側で更新されているので変更する必要はない
def adam(W, m, v, dEdW, t, 
         alpha = 0.001, beta1 = 0.9, beta2 = 0.999, tol = 10**(-8)):

    m_t = beta1 * m + (1-beta1)*dEdW
    v_t = beta2 * v + (1-beta2)*dEdW * dEdW
    
    m_hat = m_t / (1-beta1**t)
    v_hat = v_t / (1-beta2**t)
    
    W_t = W - alpha * m_hat / (np.sqrt(v_hat) + tol)

    return W_t, m_t, v_t

#####中間層のユニット数とパラメータの初期値
q = 64

W0 = np.random.normal(0, 0.3, size=(q, d+1))
W1 = np.random.normal(0, 0.3, size=(d, q+1))

########## 確率的勾配降下法によるパラメータ推定
error = []
error_test = []

num_epoch = 10

##### adamの初期値
m0 = np.zeros(shape=(q, d+1))
v0 = np.zeros(shape=(q, d+1))
m1 = np.zeros(shape=(d, q+1))
v1 = np.zeros(shape=(d, q+1))

t = 0

eta = 0.01
for epoch in range(0, num_epoch):
    index = np.random.permutation(n)

    e = np.full(n,np.nan)    
    for i in index:

        ### 課題2 ここから

        # サイズdの正規乱数を生成し，"x_train[i, :]"に足してからappendする
        noise = np.random.normal(0,0.05,d)
        # 下のx_train[i, :]に生成したノイズを足すコードに変更する: "z0 = np.append(1, x_train[i, :] + ...)"
        z0 = np.append(1, x_train[i, :] + noise)

        # 順伝播計算を作成
        # 注: W_1には変数W1, W_0には変数W0を使う
        z1, u1 = forward(z0, W0, sigmoid)
        x_tilde = np.dot(W1,z1)
               
        ##### 誤差評価
        # ErrorFunctionを作成したら以下のコメントを外す
        e[i] = ErrorFunction(x_tilde, x_train[i, :])

        if epoch == 0:
            # 誤差推移観察のepoch=0はパラメタ更新しない
            # (実際には最初から更新しても構わない)
            continue

        ### 課題2 ここまで
        
        ### 課題3 ここから

        ##### 逆伝播
        delta1 = x_tilde - x_train[i, :]
        delta0 = backward(W1[:, 1:], delta1, u1)

        ### 確率勾配降下法による更新
        # 課題3ではひとまずこちらを使う
        # 課題4ではここを再び消して，adamを定義して使う
        # eta_t = eta / epoch 
        # W1 -= eta_t*np.outer(delta1, z1)
        # W0 -= eta_t*np.outer(delta0, z0)

        ### 課題3 ここまで
        
        ### 課題4
        # adam関数の中身を実装し，以下でパラメタを更新する
        # adamで実行する場合は，
        # 課題3の確率勾配降下の更新式を消す（コメントにする）ことを忘れずに
        t += 1
        W1, m1, v1 = adam(W1, m1, v1, np.outer(delta1, z1), t)
        W0, m0, v0 = adam(W0, m0, v0, np.outer(delta0, z0), t)

    
    ##### training error
    error.append(sum(e)/n)

    e_test = np.full(n_test,np.nan)        
    ##### test error
    for j in range(0, n_test):

        #### ノイズを加えたあと順伝播 (訓練時と同じように書けばよい)
        noise = np.random.normal(0,0.05,d)
        # 下のx_test[j, :]に生成したノイズを足すコードに変更する: "z0 = np.append(1, x_test[j, :] + ...)"
        z0 = np.append(1, x_test[j, :])

        z1, u1 = forward(z0, W0, sigmoid)
        x_tilde = np.dot(W1,z1)

        #### 復元前後の画像を保存するx_tildeを保存しておくコード
        # 上でx_tildeの計算を書いたら以下２行のコメントを外す
        if epoch == (num_epoch-1) and j == index_plot_img:
           x_tilde_to_plot = np.copy(x_tilde)

        # ErrorFunctionを作成したら以下のコメントを外す        
        e_test[j] = ErrorFunction(x_tilde, x_test[j, :])
    
    error_test.append(sum(e_test)/n_test)


########## 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Squared error (log-scale)")
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", transparent=True)

if 'x_tilde_to_plot' in locals():
    D_o = np.reshape(x_test[index_plot_img, :], (28, 28))
    D = np.reshape(x_tilde_to_plot, (28, 28))

    plt.clf()
    sns.heatmap(D_o, cbar =False, cmap="Blues", square=True)
    plt.axis("off")
    plt.savefig("./original.pdf", bbox_inches='tight', transparent=True)

    plt.clf()
    sns.heatmap(D, cbar =False, cmap="Blues", square=True)
    plt.axis("off")
    plt.savefig("./reconstruct.pdf", bbox_inches='tight', transparent=True)

plt.clf()
fig,axes = plt.subplots(nrows=8,ncols=9,figsize=(10, 10))
D = np.reshape(W0[0, 1:], (28, 28)).T
axes[0,0].imshow(D)
axes[0,0].axis("off")
for i in range(8):
    axes[i,0].axis("off")
    for j in range(8):
        D = np.reshape(W0[8*i+j, 1:], (28, 28)).T
        axes[i,j+1].imshow(D)
        axes[i,j+1].axis("off")
plt.savefig("./W0.pdf", bbox_inches='tight', transparent=True)

plt.clf()
fig,axes = plt.subplots(nrows=8,ncols=9,figsize=(10, 10))
D = np.reshape(W1[:, 0], (28, 28)).T
axes[0,0].imshow(D)
axes[0,0].axis("off")
for i in range(8):
    axes[i,0].axis("off")
    for j in range(8):
        D = np.reshape(W1[:, 8*i+j+1], (28, 28)).T
        axes[i,j+1].imshow(D)
        axes[i,j+1].axis("off")
plt.savefig("./W1.pdf", bbox_inches='tight', transparent=True)
plt.close()

