import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import random
import math

# 結果に再現性を持たせるために乱数シードを固定
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

with open("./mukashi-banashi.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))  # 文字のリスト
char_size = len(chars)  # 文字の種類

char2int = {ch: i for i, ch in enumerate(chars)}  # 辞書 -> key: 文字, value: ID番号
int2char = {i: ch for i, ch in enumerate(chars)}  # 辞書 -> key: ID番号, value: 文字


# 文字列からID番号の列への変換関数
def encode(a):
    return [char2int[b] for b in a]


# ID番号から文字列への変換関数
def decode(a):
    return "".join([int2char[b] for b in a])


train_data = torch.tensor(
    encode(text), dtype=torch.long
)  # IDを並べたtensorが訓練データ

L = 8  # 入力文の文字数(今回は8で固定)
dim_embed = 128  # トークンを埋め込むベクトルの次元数
head_size = 64  # Attention layerの出力する次元(今回はクエリとキーの次元もこれを併用)
batch_size = 128  # 確率勾配降下のバッチサイズ

print("学習データで使っている文字数： ", char_size)


class PositionalEncoding(nn.Module):

    def __init__(self, dim, L):
        super().__init__()

        pe = torch.zeros(L, dim)

        # --------------------------------------------------
        for l in range(L):
            for i in range(dim_embed):
                if i % 2 == 0:
                    pe[l][i] = np.sin(l / 10000 ** (i / dim_embed))
                else:
                    pe[l][i] = np.cos(l / 10000 ** ((i - 1) / dim_embed))
        # --------------------------------------------------

        self.register_buffer("pe", pe)

    def forward(self, x):  # x: batch_size x L x dim_embed

        return x + self.pe


class SelfAttention(nn.Module):
    def __init__(self, dim_embed, L, head_size):
        super().__init__()

        self.head_size = head_size
        self.key = nn.Linear(dim_embed, head_size, bias=False)
        self.query = nn.Linear(dim_embed, head_size, bias=False)
        self.value = nn.Linear(dim_embed, head_size, bias=False)

    def forward(self, x):

        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        # --------------------------------------------------
        K_t = K.transpose(1, 2)  # Kの転置処理．Kのサイズをよく考えること

        scaled_dot_product = Q @ K_t / np.sqrt(self.head_size)

        atten_wei = F.softmax(scaled_dot_product, dim=2)  # どの次元でsoftmaxをとる?

        out = atten_wei @ V  # 最終的な出力

        return out
        # --------------------------------------------------


class Transformer(nn.Module):
    def __init__(self, dim_embed, char_size, L, head_size):
        super().__init__()

        self.token_embedding = nn.Embedding(char_size, dim_embed)

        #
        # 各層の定義を作成
        #

        self.position_encoding = PositionalEncoding(dim_embed, L)

        self.selfatten = SelfAttention(dim_embed, L, head_size)

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(in_features=head_size * L, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=char_size),
        )
        # Full connectionを指定されたMLPに変更
        # 最初の入力次元の変化に要注意(直前に出力されてるサイズ及びflattenを考えると...)

    def forward(self, idx, targets=None):

        x = self.token_embedding(idx)  # Batchsize x L x dim_embed

        #
        # Positional-encodingとSelf-attentionの実行を作成
        x = self.position_encoding(x)
        x = self.selfatten(x)
        #

        x = self.flatten(x)  # Batchsize x (L*dim_embed) の配列に変形して通常のMLPに渡す

        logits = self.fc(x)  # fcの中身を課題で指定されたものに書き換える

        return logits


# 入力された文字列から続きの文章を生成 (編集不要)
def generate_text(model, input_str, slen=50):
    if len(input_str) != L:
        print(f"Input must be {L} characters")
        return input_str

    input_idx = torch.tensor([encode(input_str)])
    output_idx = input_idx.detach().clone()
    model.eval()
    for _ in range(slen):
        logits = model(output_idx[:, -L:])  # outputの最後のL文字をモデルに
        probs = F.softmax(logits, dim=1)  # 生成確率
        idx_next_pred = torch.multinomial(probs, num_samples=1)
        output_idx = torch.cat((output_idx, idx_next_pred), dim=1)
    return decode(output_idx[0].tolist())


model = Transformer(dim_embed, char_size, L, head_size)
loss_fn = nn.CrossEntropyLoss()

print("----- 学習前 -----")
while True:
    print(f"Prompt入力(ひらがな{L}文字) > ", end="")
    input_str = input()
    if len(input_str) == L:
        break
    print(f"Input must be {L} characters")
print("NeuralNet出力 >", generate_text(model, input_str))

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
step_loss = []
for steps in range(5000):
    ix = torch.randint(
        len(train_data) - L - 1, (batch_size,)
    )  # batch_size個のスタート地点をランダムに選ぶ
    x = torch.stack(
        [train_data[i : i + L] for i in ix]
    )  # ランダム選んだ地点それぞれから長さLの文字列を取得
    y = torch.stack(
        [train_data[i + L] for i in ix]
    )  # xの最後の文字から一文字だけずれたものを予測対象に

    logits = model(x)
    loss = loss_fn(logits, y)
    step_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if np.mod(steps, 1000) == 0:
        print(f"steps = {steps}: loss = {loss.item()}")

plt.figure()
plt.plot(step_loss)
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.savefig("loss.pdf")
plt.close()


print("----- 学習後 -----")
print("入力されたプロンプト:", input_str)
print("NeuralNet出力(１回目) >", generate_text(model, input_str))
print("NeuralNet出力(２回目) >", generate_text(model, input_str))
print("NeuralNet出力(３回目) >", generate_text(model, input_str))
