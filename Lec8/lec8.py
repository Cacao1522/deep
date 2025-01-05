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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# データの読み込み
ticker = '9984.T'
data = yf.download(ticker, start='2018-01-01', end='2024-01-01')
data2 = data["Close"].values.reshape(-1, 1)

# データの前処理
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data2)

# 図で比較
plt.figure(figsize=(14, 7))

# プロット用のインデックスを作成
original_data_index = data.index.strftime('%Y-%m-%d').tolist()

# 正規化前のデータをプロット
plt.subplot(2, 1, 1)
plt.plot(original_data_index, data2.flatten(), label='Original Close')
plt.title('Original Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(original_data_index[::30], rotation=45)  # 30日ごとにラベルを表示
# 正規化後のデータをプロット
plt.subplot(2, 1, 2)
plt.plot(original_data_index, data_scaled, label='Scaled Close', color='red')
plt.title('Scaled Close Prices')
plt.xlabel('Date')
plt.ylabel('Scaled Price')
plt.legend()
plt.xticks(original_data_index[::30], rotation=45)  # 30日ごとにラベルを表示

# レイアウト調整
plt.tight_layout()
plt.savefig("price.pdf")
plt.show()

# Keras
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

# LSTMの入力の形状を (samples, time_steps, features) に変換
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTMモデルの構築
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True, input_shape=(time_step, 1))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50, return_sequences=False)))
model.add(Dense(25))
model.add(Dense(1))

# モデルのコンパイルと訓練
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=10)

# 予測
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 予測結果を逆変換
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

# モデルの評価
mse = mean_squared_error(y_test[:, 0], test_predict[:, 0])
r2 = r2_score(y_test[:, 0], test_predict[:, 0])

print('MSE: {:.2f}'.format(mse))
print('R^2: {:.2f}'.format(r2))

#プロット
plt.figure(figsize=(10, 6))
plt.plot(original_data_index, data2.flatten(), label='Historical')
train_predict_plot = np.empty_like(data_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict
plt.plot(original_data_index, train_predict_plot, label='Train Predict')

test_predict_plot = np.empty_like(data_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1, :] = test_predict
plt.plot(original_data_index, test_predict_plot, label='Test Predict')

plt.legend()
plt.xticks(original_data_index[::30], rotation=45)  # 30日ごとにラベルを表示
plt.tight_layout()
plt.savefig("priceresult.pdf")
plt.show()

# # Pytorch(精度が悪い)
# # GPUが利用可能であればGPUを使用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # データをPyTorchのテンソルに変換
# X_train = torch.from_numpy(X_train).type(torch.Tensor).to(device)
# X_test = torch.from_numpy(X_test).type(torch.Tensor).to(device)
# y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
# y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
#         super(LSTM, self).__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=1, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(0.2)
#         self.linear = nn.Linear(hidden_layer_size * 2, output_size)
#         self.hidden_cell = (torch.zeros(2, 1, self.hidden_layer_size).to(device),
#                             torch.zeros(2, 1, self.hidden_layer_size).to(device))
#     def forward(self, input_seq):
#         h0 = torch.zeros(2, input_seq.size(0), self.hidden_layer_size).to(device)
#         c0 = torch.zeros(2, input_seq.size(0), self.hidden_layer_size).to(device)
#         lstm_out, _ = self.lstm(input_seq, (h0, c0))
#         lstm_out = self.dropout(lstm_out)
#         predictions = self.linear(lstm_out[:, -1])
#         return predictions
#     # def forward(self, input_seq):
#     #     lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
#     #     lstm_out = self.dropout(lstm_out)
#     #     predictions = self.linear(lstm_out[:, -1])
#     #     return predictions

# model = LSTM().to(device)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     for i in range(len(X_train)):
#         optimizer.zero_grad()
#         X_batch = X_train[i].unsqueeze(0)  # バッチサイズを1に設定
#         y_batch = y_train[i].unsqueeze(0)  # バッチサイズを1に設定
#         y_pred = model(X_batch)
#         single_loss = loss_function(y_pred, y_batch)
#         single_loss.backward()
#         optimizer.step()
    
#     if epoch % 1 == 0:
#         print(f'Epoch {epoch+1} Loss: {single_loss.item()}')

# model.eval()
# train_predict = model(X_train).cpu().detach().numpy()
# test_predict = model(X_test).cpu().detach().numpy()

# # 予測結果を逆変換
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)
# y_test = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

# # モデルの評価
# mse = mean_squared_error(y_test, test_predict)
# r2 = r2_score(y_test, test_predict)

# print('MSE: {:.2f}'.format(mse))
# print('R^2: {:.2f}'.format(r2))

# # プロット用のインデックスを修正
# plt.figure(figsize=(10, 6))
# plt.plot(original_data_index, data2.flatten(), label='Historical')
# train_predict_plot = np.empty_like(data_scaled)
# train_predict_plot[:, :] = np.nan
# train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict
# plt.plot(original_data_index, train_predict_plot, label='Train Predict')

# test_predict_plot = np.empty_like(data_scaled)
# test_predict_plot[:, :] = np.nan
# test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1, :] = test_predict
# plt.plot(original_data_index, test_predict_plot, label='Test Predict')

# plt.legend()
# plt.xticks(original_data_index[::30], rotation=45)
# plt.tight_layout()
# plt.savefig("priceresult.pdf")
# plt.show()
