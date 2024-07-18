
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import random

# 結果に再現性を持たせるために乱数シードを固定
random.seed(100) 
np.random.seed(100)
torch.manual_seed(100)

chars = [chr(i) for i in range(ord("a"), ord("z")+1)] # 文字のリスト
char_size = len(chars) # 文字の種類

# 文字とIDの変換辞書
char2int = { ch : i for i, ch in enumerate(chars) } # key: 文字 => value: ID番号
int2char = { i : ch for i, ch in enumerate(chars) } # key: ID番号 => value: 文字 

# 文字列からID番号の列への変換関数
def encode(a):
    return [char2int[b] for b in a]
# ID番号から文字列への変換関数
def decode(a):
    return ''.join([int2char[b] for b in a])

dim_embed = 2 # 埋め込み次元

class PositionalEncoding(nn.Module): 

    def __init__(self, dim_embed, L):
        # dim_embed: トークンを埋め込むベクトルの次元数, L: 入力文の文字数
        super().__init__()

        pe = torch.zeros(L, dim_embed) # L x dim_embed 配列にPositional Encodingを作成

        #
        # ここにpeの要素を定義するコードを作成
        for l in range(L):
            for i in range(dim_embed):
                if (i%2 == 0):
                    pe[l][i] = np.sin(l/10000**(i/L))
                else:
                    pe[l][i] = np.cos(l/10000**((i-1)/L))
        #     
        
        self.register_buffer('pe', pe) # 今回のPositionalEncodingは学習しない定数であることの指定(これ以降，クラス内からはself.peでアクセス可能(編集不要)

    def forward(self, x): # x: batch_size x L x dim_embed

        return x + self.pe # xとpeのサイズを考慮するとこの足し算がどうなるか考えてみること(参考: 課題1(d))


class SelfAttention(nn.Module):

    def __init__(self, dim_embed, head_size):
        # dim_embed: トークンを埋め込むベクトルの次元数, L: 入力文の文字数, head_size: Attention layerの出力する次元(今回はクエリとキーの次元もこれを併用)
        
        super().__init__()

        self.head_size = head_size
        self.key = nn.Linear(dim_embed, head_size, bias=False) 
        self.query = nn.Linear(dim_embed, head_size, bias=False)
        self.value = nn.Linear(dim_embed, head_size, bias=False)

    def forward(self, x):
        
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        print(K)
        #
        # ここにK, Q, Vを使ったself-attentionの処理を作成
        # 
        
        K_t = K.T#Kの転置処理．Kのサイズをよく考えること
        
        scaled_dot_product = (Q @ K_t / np.sqrt(self.head_size))@V

        atten_wei = F.softmax(scaled_dot_product, dim=0) # どの次元でsoftmaxをとる?
        
        # out = ... 最終的な出力

        # return out

def plot_embedding(strs, pos, fname):
    plt.figure()
    cols = ['r','b','g']
    mrk = ['o','x','+']
    for j in range(len(strs)):
        plt.scatter(pos[j,:,0], pos[j,:,1], c=cols[j], marker=mrk[j],label=strs[j])
        for i in range(len(strs[j])):
            plt.text(pos[j,i,0], pos[j,i,1], strs[j][i] + "(" + str(i) + ")", c=cols[j])
    plt.legend()
    plt.savefig(fname)
    plt.close()

print("Character index list")
for c in chars:
    print(f"{c}: {encode(c)}", end=" ")    
print("")

token_embedding = nn.Embedding(char_size, dim_embed)    

pe = PositionalEncoding(dim_embed, 6)

strs = ["tiktok","tictac","tuktuk"]

# 文字列をindexの数値列に変換
input_idx = torch.tensor([encode(s) for s in strs])

# 文字列と数値列の対応関係
for i in range(len(strs)):
    print(f"{strs[i]}: {input_idx[i]}")

x = token_embedding(input_idx)
print(x)
plot_embedding(strs, x.detach().numpy(), 'embed-1.pdf') # x.detach().numpy()はxをnumpy配列に変換

# -- 以降は実装したら順次コメントを外す --

x = pe(x)

plot_embedding(strs, x.detach().numpy(), 'embed-2.pdf')        

# selfatten = SelfAttention(dim_embed=2, head_size=2)

# x = selfatten(x)

# plot_embedding(strs, x.detach().numpy(), 'embed-3.pdf')
