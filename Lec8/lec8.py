import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
ticker = '9984.T'
data = yf.download(ticker, start='2021-01-01', end='2024-01-01')
data.columns
data = data["Close"]

# データの前処理
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

# 図で比較
plt.figure(figsize=(14, 7))

# 正規化前のデータをプロット
plt.subplot(2, 1, 1)
plt.plot(data.index, data, label='Original Close')
plt.title('Original Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 正規化後のデータをプロット
plt.subplot(2, 1, 2)
plt.plot(data.index, data_scaled, label='Scaled Close', color='red')
plt.title('Scaled Close Prices')
plt.xlabel('Date')
plt.ylabel('Scaled Price')
plt.legend()

# レイアウト調整
plt.tight_layout()
#plt.show()
plt.savefig("price.pdf")

# 訓練データとテストデータの分割
training_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:training_size], data_scaled[training_size:]

# データをLSTM用に変換
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
print("通過１")
# LSTMの入力の形状を (samples, time_steps, features) に変換
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTMモデルの構築
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
print("通過２")
# モデルのコンパイルと訓練
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)
print("通過３")
# 予測
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
print("通過４")
# 予測結果を逆変換
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])
print("通過５")
# モデルの評価
mse = mean_squared_error(y_test[0], test_predict[:, 0])
r2 = r2_score(y_test[0], test_predict[:, 0])

print('MSE: {:.2f}'.format(mse))
print('R^2: {:.2f}'.format(r2))

# Output
# MSE: 87.30
# R^2: -0.08

# プロット
plt.figure(figsize=(10, 6))
plt.plot(data.index, data, label='Historical')
train_predict_plot = np.empty_like(data_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict
plt.plot(data.index, train_predict_plot, label='Train Predict')

test_predict_plot = np.empty_like(data_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1, :] = test_predict
plt.plot(data.index, test_predict_plot, label='Test Predict')

plt.legend()
plt.show()

# # サンプルデータ
# data = torch.tensor([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12]
# ])

# targets = torch.tensor([
#     4, 8, 12
# ])


# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()

#         self.lstm = nn.LSTM(input_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         output, _ = self.lstm(x)
#         output = output[-1]  # 最後の隠れ状態を使用
#         output = self.fc(output)
#         return output
    
    

# model = LSTMModel(data.shape[1], 16, 1)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters())

# for epoch in range(10):
#     for i in range(data.shape[0]):
#         x = data[i].view(1, -1)  # バッチサイズ1のデータ
#         y = targets[i]

#         optimizer.zero_grad()
#         output = model(x)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch {epoch+1}, Step {i+1}: loss = {loss.item():.4f}")
        
# new_data = torch.tensor([13, 14, 15, 16])
# prediction = model(new_data.view(1, -1))
# print(f"Prediction: {prediction.item():.4f}")