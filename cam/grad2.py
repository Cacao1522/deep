import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. 事前学習済みモデルのロード
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # 推論モード

# 最終畳み込み層の指定
target_layer = model.layer4[-1]

# 2. ImageNetクラスラベルの取得
import requests

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(url).json()

# 3. Grad-CAM のクラス
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # フォワードフック
        self.target_layer.register_forward_hook(self.save_features)
        # バックワードフック
        self.target_layer.register_full_backward_hook(self.save_gradients)

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
    
class RelevanceCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None

        # フォワードフック
        self.target_layer.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.features = output

    def __call__(self, input_tensor):
        # フォワードパス
        _ = self.model(input_tensor)

        # 活性化マップを直接利用
        cam = self.features.mean(dim=1, keepdim=True)  # チャネルごとに平均を計算
        cam = torch.nn.functional.relu(cam)  # ReLUを適用

        return cam.squeeze().detach().cpu().numpy()
# 4. 入力画像の前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 画像の読み込み
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    return image, input_tensor

# 5. Grad-CAM の適用と可視化
def visualize_cam(image, cam, name):
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
    #plt.show()
    plt.savefig(image_path+name, bbox_inches="tight", pad_inches=0)
    plt.close()
# Grad-CAMインスタンスの生成
#grad_cam = GradCAM(model, target_layer)
grad_cam = RelevanceCAM(model, target_layer)
# 入力画像のパス
image_path = "dogcat1"  # 例: テスト画像パス
image, input_tensor = load_image(image_path + ".jpg")

# 推論と可視化
input_tensor = input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
output = model(input_tensor)
_, predicted_class = output.max(1)
class_idx = predicted_class.item()

print(f"Predicted Class: {labels[class_idx]}")  # 予測クラス名

# Grad-CAMの生成と可視化
#cam = grad_cam(input_tensor, class_idx)
#visualize_cam(image, cam, "g")
cam = grad_cam(input_tensor)
visualize_cam(image, cam, "r")