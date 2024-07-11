import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import random
import numpy as np

# 結果に再現性を持たせるために乱数シードを固定
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

trainset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
testset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

max_epoch = 5
batch_size = 100
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #
        # 自由に作成 (Epochやoptimizerなどその他の場所も自由に編集可)
        #
        self.fe = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=24, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),  
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24), 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24), 
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=1224, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),  # Batch Normalization
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=24 * 14 * 14, out_features=2048),
            #nn.BatchNorm1d(2048),  
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=10),
        )

    def forward(self, x):

        #
        # 自由に作成 (Epochやoptimizerなどその他の場所も自由に編集可)
        #
        #print(x.size())
        x = self.fe(x)
        #print(x.size())
        logits = self.fc(x)
        return logits


model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6
)


def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
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


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            out = model(X)
            pred_label = out.argmax(1)
            correct += (pred_label == y).type(torch.float).sum().item()
            loss = loss_fn(out, y)
            test_loss += loss.item()

    correct /= size
    test_loss /= len(dataloader)

    return test_loss, correct


log_train_loss = []
log_test_loss = []
log_test_correct = []
for epoch in range(max_epoch):
    print("Epoch = ", epoch)

    train_loss = train_loop(train_loader, model, loss_fn, optimizer, scheduler)
    test_loss, test_correct = test_loop(test_loader, model, loss_fn)
    print(f" Test Accuracy: {(100*test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    scheduler.step(test_loss)
    log_train_loss.append(train_loss)
    log_test_loss.append(test_loss)
    log_test_correct.append(test_correct)

print("Final Accuracy: {:.1f} %".format(100 * test_correct))

plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(log_train_loss, label="Training", lw=3, c="b")
ax1.set_xlabel("Epoch", fontsize=18)
ax1.set_ylabel("Train loss", fontsize=18, color="b")
ax2.plot(log_test_correct, lw=3, c="r")
ax2.set_ylabel("Test accuracy", fontsize=18, color="r")
plt.tight_layout()
plt.savefig("./error.pdf")
