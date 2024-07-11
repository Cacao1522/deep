import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import sys

trainset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)


class CNN(nn.Module):  # nn.Moduleを継承したCNNクラス
    def __init__(self):  # 初期化．使用する層の定義
        super(CNN, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=6 * 13 * 13, out_features=10
        )  # 10classに対応する10次元出力．softmax関数は損失関数側で計算される

    def forward(self, x):
        print("1:", x.shape)
        x = self.conv(x)
        print(x.size())
        sys.exit()
        x = self.relu(x)
        print(x.shape)
        x = self.maxpooling(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        logits = self.linear(x)
        print(logits.shape)
        return logits


model = CNN()  # CNNクラスのオブジェクト生成
train_loader = DataLoader(trainset, batch_size=100, shuffle=True)
loss_fn = nn.CrossEntropyLoss()  # クロスエントロピー損失(softmax関数含む)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()  # Trainモード (validationやtestではmodel.eval()とする)
    running_loss = 0.0
    for X, y in dataloader:
        out = model(X)  # モデルの出力
        loss = loss_fn(out, y)  # 損失関数の計算
        running_loss += loss.item()  # 損失関数を積算

        loss.backward()  # 勾配計算
        optimizer.step()  # パラメタ更新
        optimizer.zero_grad()  # 勾配のリセット

    return running_loss / len(dataloader)


max_epoch = 5
for epoch in range(max_epoch):  # 学習のmain loop
    print("Epoch = ", epoch)

    train_loss = train_loop(train_loader, model, loss_fn, optimizer)
    print("  train_loss =", train_loss)

testset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)  # Test dataの読み込み
test_loader = DataLoader(testset, batch_size=100, shuffle=False)

model.eval()  # 評価モードに切り替え
test_loss = 0
correct = 0
with torch.no_grad():  # 勾配計算をしないモード
    for X, y in test_loader:
        out = model(X)
        pred_label = out.argmax(dim=1)  # 10次元のoutput中の最大値の次元
        correct += (pred_label == y).type(torch.float).sum().item()  # 正解した数を足す

correct /= len(test_loader.dataset)  # データ数で割って正解率を計算
print(f"Accuracy: {(100*correct):>0.1f}%")
