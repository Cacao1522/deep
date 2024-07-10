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

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

max_epoch = 5 
batch_size = 100
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fe = nn.Sequential(
            # ???
        )
        self.fc = nn.Sequential(
            # ???
        )

    def forward(self, x):
        # ???

model = CNN()
loss_fn = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), 
                       lr=0.01, betas=(0.9, 0.999), eps=1e-08)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train() # Trainモード (validationやtestではmodel.eval()とする)
    running_loss = 0.0
    for X, y in dataloader:
        out = model(X) # モデルの出力
        loss = loss_fn(out, y) # 損失関数の計算
        running_loss += loss.item() # 損失関数を積算
        
        loss.backward() # 勾配計算
        optimizer.step() # パラメタ更新
        optimizer.zero_grad() # 勾配のリセット

    return running_loss / len(dataloader)    

def test_loop(dataloader, model, loss_fn): 
    model.eval() # Evalモード
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

    train_loss = train_loop(train_loader, model, loss_fn, optimizer)    
    test_loss, test_correct = test_loop(test_loader, model, loss_fn)
    print(f" Test Accuracy: {(100*test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")        
    log_train_loss.append(train_loss) 
    log_test_loss.append(test_loss)
    log_test_correct.append(test_correct)     

plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(log_train_loss, label="Training", lw=3, c='b')
ax1.set_xlabel("Epoch", fontsize=18)
ax1.set_ylabel("Train loss", fontsize=18, color='b')
ax2.plot(log_test_correct, lw=3, c='r')
ax2.set_ylabel("Test accuracy", fontsize=18, color='r')
plt.tight_layout()
plt.savefig("./error.pdf")
