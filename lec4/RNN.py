#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4回演習問題
"""
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy.io import loadmat

np.random.seed(0)

##### データセットの準備

# 1.ブラウザ
# https://github.com/dsiufl/Reservoir-Computing　にアクセスして緑色のCodeボタンを押し，zipファイルをダウンロードする．
# これを解凍して，フォルダ"Lyon_decimation_128"を作業フォルダ直下に移動する．

# 2. コマンドプロンプト
# > git clone https://github.com/dsiufl/Reservoir-Computing.git
# フォルダ"Reservoir-Computing"ができるので，その中のフォルダ"Lyon_decimation_128"を作業フォルダ直下に移動する．

##### データ読み込み用関数
def read_data(dir_name, utterance_train, utterance_test, scaling_factor):
    ''' 
    :入力：
    : dir_name: データファイル(.mat)の入っているディレクトリ名
    : utterance_train: 訓練用のutteranceのインデックスリスト
    : utterance_test:  テスト用のutteranceのインデックスリスト
    : scaling_factor: データのスケーリングファクタ
    :出力：
    : x_train, y_train: 訓練用の入力データ(x_train)と正解ラベル(y_train)
    : x_test, y_test: テスト用の入力データ(y_train)と正解ラベル(y_test)
    '''
    # .matファイルのみを取得
    data_files = glob.glob(os.path.join(dir_name, '*.mat'))
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    # データ読み込み
    if len(data_files) > 0:
        print("%d files in %s を読み込んでいます..." % (len(data_files), dir_name))
        for each_file in data_files:
            data = loadmat(each_file)
            utterance = int(each_file[-8])  # 発話インデックス
            digit = int(each_file[-5]) # 発話数字
            if utterance in utterance_train:  # 訓練用
                # 入力データ（構造体'spec'に格納されている）
                x_train.append(data['spec'].T*scaling_factor)  # (Time x Channel)
                # 正解ラベル
                y_train.append(digit)
            elif utterance in utterance_test:  # テスト用
                # 入力データ（構造体'spec'に格納されている）
                x_test.append(data['spec'].T*scaling_factor)  # (Time x Channel)
                # 正解ラベル
                y_test.append(digit)
    else:
        print("ディレクトリ %s にファイルが見つかりません．" % (indir))
        return
    return x_train, y_train, x_test, y_test


##### データ読み込み
dir_name = './Lyon_decimation_128'  # データが保存されているディレクトリ名
d = 77  # 周波数チャネル数
m = 10  # クラス数(0-9の数字)
scaling_factor = 1e+3  # スケーリングパラメータ
#utterance_train = list(range(1,8))  # 訓練用インデックス(作成時は少なく設定しておくとよい)
utterance_train = [1,2,3,4,5,6,7]  # 訓練用インデックス(作成時は少なく設定しておくとよい)
#utterance_test = list(range(8,11))  # テスト用インデックス
utterance_test = [8,9,0]  # テスト用インデックス
x_train, y_train, x_test, y_test = read_data(dir_name, utterance_train, utterance_test, scaling_factor)

n_train = len(x_train)  # 訓練データ数
n_test = len(x_test)  # テストデータ数

# ラベルをone-hotベクトルに変換
y_train_vec = []
for label in y_train:
    vec = np.zeros(m)
    vec[label] = 1
    y_train_vec.append(vec)

y_test_vec = []
for label in y_test:
    vec = np.zeros(m)
    vec[label] = 1
    y_test_vec.append(vec)
    
# plot_misslabeled = True
plot_misslabeled = False

##### 活性化関数, 誤差関数, 順伝播, 逆伝播
def softmax(x):
    u = x.T
    e = np.exp(u-np.max(u, axis=0))
    return (e/np.sum(e, axis = 0)).T

def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp, tmp*(1-tmp)

def ReLU(x):
    return x*(x>0), 1*(x>0)
  
def Tanh(x):
    # ハイパボリックタンジェントとその微分を返す関数を作成

    return np.tanh(x),1 - np.tanh(x)**2

def CrossEntoropy(x, y):
    # returnの後にクロスエントロピーを返すプログラムを書く
    # epsilon = 1e-12  # 小さな定数を追加
    # x = np.clip(x, epsilon, 1. - epsilon)
    return -np.sum(y*np.log(x))

def forward(x, z_prev, W_in, W, actfunc):
    ### 課題1. 順伝播のプログラムを書く
    # 注意: xは呼び出し元で定数項も含む形で渡されている
    # 注意: z_prevはz_{t-1}に対応
    # 注意: actfuncは一つ目の返り値として活性化関数fの値を，
    #       二つ目の返り値として活性化関数を微分したnabla fの値を返す
    tmp = np.dot(W_in, x)+np.dot(W, z_prev) #
    z = actfunc(tmp)[0] # 
    u = actfunc(tmp)[1] #
    return z, u #

def backward(W, W_out, delta, delta_out, derivative):
    # 逆伝播のプログラムを書く
    return (np.dot(W.T, delta)+np.dot(W_out.T, delta_out))*derivative

def adam(W, m, v, dEdW, t, 
         alpha = 0.001, beta1 = 0.9, beta2 = 0.999, tol = 10**(-8)):
    m_t = beta1*m+(1-beta1)*dEdW
    v_t = beta2*v+(1-beta2)*dEdW**2
    
    m_hat = m_t/(1-beta1**t)
    v_hat = v_t/(1-beta2**t)
    
    w_t = W - alpha*m_hat/(np.sqrt(v_hat)+tol)
    return w_t, m_t, v_t

##### 中間層のユニット数とパラメータの初期値
q = 128

W_in = np.random.normal(0, 0.2, size=(q, d+1))
W = np.random.normal(0, 0.2, size=(q, q))
W_out = np.random.normal(0, 0.2, size=(m, q+1))

########## 確率的勾配降下法によるパラメータ推定
# num_epoch = 40
num_epoch = 30

error = []
error_test = []

prob = np.zeros((n_test,m))

##### adamのパラメータの初期値
m_in = np.zeros(shape=W_in.shape)
v_in = np.zeros(shape=W_in.shape)
m_hidden = np.zeros(shape=W.shape)
v_hidden = np.zeros(shape=W.shape)
m_out = np.zeros(shape=W_out.shape)
v_out = np.zeros(shape=W_out.shape)

eta = 0.01

n_update = 0

print("学習を行っています...")
for epoch in range(0, num_epoch):
    index = np.random.permutation(n_train)
    print("epoch =",epoch)

    e = np.full(n_train, np.nan)        
    for i in index:
        xi = x_train[i]  # xiは行列（時間幅T×チャネル数d）
        yi = y_train_vec[i]  # yiはラベルのone-hotベクトル
        T = xi.shape[0] 
        
        ##### 順伝播
        # 課題1. Z_prime, nabla_fを作成する
        Z_prime = np.zeros((q,T+1))
        nabla_f = np.zeros((q,T))

        for t in range(T):
            # Z_primeの「t+1列目」，nabla_fの「t列目」をforwardを使って求める
            Z_prime[:,t+1], nabla_f[:,t] = forward(np.append(1, xi[t,:]), Z_prime[:,t], W_in, W, sigmoid)
        
        Z_T = np.append(1, Z_prime[:,T])

        z_out = softmax(np.dot(W_out, Z_T))        

        ##### 誤差評価
        e[i] = CrossEntoropy(z_out, yi)

        if epoch == 0:
            # 誤差推移観察のepoch=0はパラメタ更新しない
            # (実際には最初から更新しても構わない)
            continue
        
        ##### 課題2. 逆伝播
        # delta_outを定義する
        delta_out = z_out - yi

        # 以下の行列の各列にdelta_1, ..., delta_Tを作成
        # backward関数の内部を作成
        delta = np.zeros((q,T)) 
        for t in reversed(range(T)):
            if t == T-1:
                delta[:,t] = backward(W, W_out[:,1:], np.zeros(q), delta_out, nabla_f[:,t]) 
            else:        
                delta[:,t] = backward(W, W_out[:,1:], delta[:,t+1], np.zeros(m), nabla_f[:,t]) 
        
        ##### 課題3. 勾配の計算

        ## dEdW_outの作成
        # ヒント: np.dotかnp.outerのどちらを使うべきか適切に判断すること
        #         また，上で作成したZ_Tを利用できる
        dEdW_out = np.outer(delta_out,Z_T)
        
        ## dEdE_inの作成
        # ヒント: 以下のXが定数項含んだTx(d+1)行列 
        # (np.c_は横方向の結合. Xをコンソールで見てみると
        #  何が行われいてるかわかってよい)
        X = np.hstack((np.ones(T).reshape(-1,1), xi))
        # print(delta.shape)
        # print(X.shape)
        dEdW_in = np.dot(delta,X)

        ## dEdWの作成
        # ヒント: Z_primeの0列目からT-1列目(つまり最後の列以外)は"Z_prime[:,:T]"で指定できる
        #         また，転置の存在に注意せよ
        dEdW = np.dot(delta,Z_prime[:,:T].T)

        ##### パラメータの更新
        # W_out -= eta*dEdW_out/epoch
        # W -= eta*dEdW/epoch
        # W_in -= eta*dEdW_in/epoch

        ##### 課題4. adamを作成して更新方法を以下に変更（上の確率勾配降下の更新は消す）
        n_update += 1
        W_out, m_out, v_out = adam(W_out, m_out, v_out, dEdW_out, n_update)
        W, m_hidden, v_hidden = adam(W, m_hidden, v_hidden, dEdW, n_update)
        W_in, m_in, v_in = adam(W_in, m_in, v_in, dEdW_in, n_update)

    ##### training error
    error.append(sum(e)/n_train)

    e_test = np.full(n_test,np.nan)            
    ##### test error
    for i in range(0, n_test):
        xi = x_test[i]  # xiは行列（時間幅×チャネル数）
        yi = y_test_vec[i]  # yiはラベルのone-hotベクトル
        T = xi.shape[0] 
        
        ##### 順伝播
        Z_prime = np.zeros((q,T+1))
        for t in range(T):
        # 訓練の時と同じ手順でZ_primeを作成
        # (こちらではnabla_fは使用しないので, 最後に"[0]"を
        #  つけることで返り値を一つだけ受け取っている)
            Z_prime[:,t+1] = forward(np.append(1, xi[t,:]), Z_prime[:,t], W_in, W, sigmoid)[0]
            
        z_out = softmax(np.dot(W_out, np.append(1, Z_prime[:,T])))        
        prob[i,:] = z_out

        e_test[i] = CrossEntoropy(z_out, yi)
    
    error_test.append(sum(e_test)/n_test)

########## 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)  #青線
plt.plot(error_test, label="test", lw=3)  #オレンジ線
# plt.yscale("log")
plt.xlabel("Epoch", fontsize=18)
# plt.ylabel("Cross-entropy (log-scale)",fontsize=18)
plt.ylabel("Cross-entropy",fontsize=18)
plt.grid()
plt.legend(fontsize = 16)
plt.savefig("./error.png", bbox_inches='tight', transparent=True)

predict = np.argmax(prob, 1)

if plot_misslabeled:
    n_maxplot = 20
    n_plot = 0

    ##### 誤分類結果のプロット

    for i in range(m):
        idx_true = (y_test[:, i]==1)
        for j in range(m):
            idx_predict = (predict==j)
            # ConfMat[i, j] = sum(idx_true*idx_predict)
            if j != i:
                for l in np.where(idx_true*idx_predict == True)[0]:
                    plt.clf()
                    D = x_test[l, :, :]
                    sns.heatmap(D, cbar =False, cmap="Blues", square=True)
                    plt.axis("off")
                    plt.title('{} to {}'.format(i, j))
                    plt.savefig("./misslabeled{}.png".format(l), bbox_inches='tight', transparent=True)
                    n_plot += 1
                    if n_plot >= n_maxplot:
                        break
            if n_plot >= n_maxplot:
                break
        if n_plot >= n_maxplot:
            break

predict_label = np.argmax(prob, axis=1)
true_label = np.argmax(y_test_vec, axis=1)


ConfMat = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        ConfMat[i, j] = np.sum((true_label == i) & (predict_label == j))

plt.clf()
fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="1", cbar =False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.png", bbox_inches="tight", transparent=True)
plt.close()

