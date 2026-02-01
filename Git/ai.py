import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from PIL import Image

# --- 1. 必須定義與訓練時完全一樣的模型結構 ---
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

# --- 2. 視訊主程式 ---
def run_realtime_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # CIFAR-10 的十個類別
    classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    # 載入模型
    print("正在載入模型：resnet50_224_extreme.pth...")
    model = ResNet50().to(device)
    # 使用 weights_only=True 是安全的好習慣
    checkpoint = torch.load('resnet50_extreme.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 定義與訓練時相同的影像預處理
    inference_transform = transforms.Compose([
        transforms.Resize((32, 32)), # 必須縮小到 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 啟動視訊鏡頭
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤：找不到鏡頭！")
        return

    print("啟動成功！請按 'Q' 鍵退出。")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 轉換 BGR 到 RGB，並轉為 PIL Image 格式進行 transform
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 預處理並送入 GPU
        input_tensor = inference_transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            # 取得信心度（機率）
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
            label = classes[pred.item()]
            score = conf.item()

        # 顯示結果在視窗上 (僅在信心度大於 50% 時顯示，避免雜訊亂跳)
        color = (0, 255, 0) if score > 0.7 else (0, 255, 255)
        text = f"{label}: {score:.2%}"
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 顯示視窗
        cv2.imshow('RTX 4050 Real-time ResNet50', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime_inference()