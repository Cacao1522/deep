import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights

# # ディレクトリの設定
# original_dir = "data/train"  # 元のデータが格納されているディレクトリ
# train_dir = "data/train_split"  # 新しいトレーニング用ディレクトリ
# val_dir = "data/val"  # 検証用ディレクトリ

# # クラス（フォルダ）ごとに処理
# for category in ["cat", "dog"]:
#     # 元データのパスを取得
#     category_dir = os.path.join(original_dir, category)
#     images = os.listdir(category_dir)

#     # トレーニング用と検証用に分割
#     train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

#     # 出力フォルダを準備
#     os.makedirs(os.path.join(train_dir, category), exist_ok=True)
#     os.makedirs(os.path.join(val_dir, category), exist_ok=True)

#     # 画像を移動
#     for image in train_images:
#         shutil.copy(os.path.join(category_dir, image), os.path.join(train_dir, category, image))
#     for image in val_images:
#         shutil.copy(os.path.join(category_dir, image), os.path.join(val_dir, category, image))

# print("データ分割が完了しました。")

# 入力画像の前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# データセットの作成
train_dataset = ImageFolder(root="data/train_split", transform=transform)
val_dataset = ImageFolder(root="data/val", transform=transform)

# データローダー
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# モデルのロードと出力層の調整（クラス数を指定）
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_classes = 2  # 動物クラス数（例: "cat" と "dog"）
model.fc = nn.Linear(model.fc.in_features, num_classes)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 再学習用の損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# モデルの再学習
for epoch in range(10):  # エポック数を設定
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 検証
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}, Validation Accuracy: {correct / total:.2f}")
    
    

model.eval()  # 推論モード
target_layer = model.layer4[-1]  # ResNetの場合の最終畳み込み層
# # 入力画像の読み込みと前処理
# image = Image.open("input.jpg").convert("RGB")
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(image).unsqueeze(0)  # バッチ次元を追加

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # フォワードフック
        self.target_layer.register_forward_hook(self.save_features)
        # バックワードフック
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, class_idx):
        # 勾配の平均を計算
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.features).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)  # ReLUを適用

        return cam.squeeze().detach().numpy()

    def __call__(self, input_tensor, target_class):
        # フォワードパス
        output = self.model(input_tensor)
        self.model.zero_grad()

        # ターゲットクラスに対する勾配を計算
        target = output[0, target_class]
        target.backward()

        return self.generate_cam(target_class)

# Grad-CAMインスタンスの生成
grad_cam = GradCAM(model, target_layer)


# # 推論したいクラス（ImageNetのクラスID）
# target_class = 243  # "bull mastiff" など
# cam = grad_cam(input_tensor, target_class)



# 画像を読み込み、分類
def classify_and_visualize(image_path, grad_cam, model, classes):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 推論
    output = model(input_tensor)
    _, predicted_class = output.max(1)
    class_idx = predicted_class.item()

    # Grad-CAM生成
    cam = grad_cam(input_tensor, class_idx)

    return image, cam, classes[class_idx]

# クラス名リスト
classes = ["cat", "dog"]  # 必要に応じて変更

def visualize_cam(image, cam):
    # CAMを正規化
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    # CAMを元画像サイズにリサイズ
    cam_image = Image.fromarray(cam).resize(image.size, Image.BILINEAR)

    # ヒートマップを重ねる
    plt.imshow(image)
    plt.imshow(cam_image, cmap="jet", alpha=0.5)  # 重ね表示
    plt.axis("off")
    plt.show()

# 入力画像のパス
image_path = "dog3.jpg"  # 例: テスト画像パス

# 分類とGrad-CAMの可視化
image, cam, predicted_class = classify_and_visualize(image_path, grad_cam, model, classes)
print(f"Predicted Class: {predicted_class}")
visualize_cam(image, cam)

# # CAMを元画像サイズにリサイズ
# cam = cam - np.min(cam)
# cam = cam / np.max(cam)
# cam = np.uint8(255 * cam)
# cam = Image.fromarray(cam).resize(image.size, Image.BILINEAR)

# # ヒートマップ表示
# plt.imshow(image)
# plt.imshow(cam, cmap='jet', alpha=0.5)  # 元画像と重ね合わせ
# plt.axis('off')
# plt.show()