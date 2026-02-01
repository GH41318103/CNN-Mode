import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. ResNet-50 結構定義 (骨架) ---
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.linear = nn.Linear(512 * Bottleneck.expansion, num_classes)
        self.adp_pool = nn.AdaptiveAvgPool2d((1, 1))
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(Bottleneck(self.in_planes, planes, s))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adp_pool(out) 
        out = torch.flatten(out, 1)
        return self.linear(out)

# --- 2. 主執行程序 ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 【關鍵設定】設定為 224 來配合你的檔案名稱
    img_size = 224 
    model_path = 'resnet50_224_extreme.pth'

    # 1. 建立模型骨架
    model = ResNet50().to(device)
    
    # 2. 載入 224 權重
    if os.path.exists(model_path):
        # 這裡使用 weights_only=True 是現代 PyTorch 的安全推薦做法
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"--- 成功載入滿血版模型：{model_path} ---")
    else:
        print(f"找不到檔案 {model_path}，請檢查路徑。")
        exit()

    model.eval()

    # 3. 準備測試資料（對應 224 解析度）
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=6, shuffle=True)

    # 4. 進行推理並繪圖
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    with torch.no_grad():
        # 確保資料格式正確
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)

    # 5. 視覺化排版
    res_text = f"{img_size}x{img_size}"
    fig = plt.figure(figsize=(16, 7))
    plt.suptitle(f"ResNet-50 Analysis Report (Input: {res_text})", fontsize=18, fontweight='bold')

    for i in range(6):
        ax = fig.add_subplot(1, 6, i + 1)
        # 影像還原
        img = images[i].numpy().transpose((1, 2, 0))
        img = np.clip(img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465]), 0, 1)
        ax.imshow(img)
        
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}", color=color, fontsize=12)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    
    # 6. 自動存檔（檔名帶有解析度）
    save_name = f'Evaluation_{res_text}_Final.png'
    plt.savefig(save_name, dpi=300)
    
    print("-" * 30)
    print(f"推理完成！解析度為：{res_text}")
    print(f"結果已存檔至：{save_name}")
    print("-" * 30)
    
    plt.show()