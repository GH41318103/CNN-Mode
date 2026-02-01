import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

# --- 1. 硬體與顯存深度優化設定 ---
warnings.filterwarnings('ignore', category=UserWarning)

# 【核心】解決 RTX 40 系列顯存碎片化問題，防止 OOM 關鍵
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 針對 Ada Lovelace 架構優化
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

# --- 2. 定義 ResNet-50 結構 ---
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

# --- 3. 執行主體 ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 核心參數：針對 6GB 顯存的最佳化平衡
    num_epochs = 15 
    img_size = 80      # 降至 80 可大幅減少 Activations 佔用的空間
    batch_size = 64    # 64 是 6GB 顯卡跑 ResNet-50 的安全上限

    # 資料預處理
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True
    )

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 建立模型並轉換為 Channels Last (提高 RTX 40 系列效率)
    model = ResNet50().to(device, memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=num_epochs)
    scaler = torch.amp.GradScaler('cuda')

    history = {'loss': [], 'acc': []}

    print(f"--- 啟動 RTX 4050 協同優化模式 (Size: {img_size}, Batch: {batch_size}) ---")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        # 【釋放機制】每個 Epoch 開始前清空快取
        torch.cuda.empty_cache()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            
            # 【釋放機制】使用 set_to_none 直接物理回收顯存
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        history['loss'].append(train_loss/(batch_idx+1))
        history['acc'].append(acc)
        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {history["loss"][-1]:.3f} | Acc: {acc:.2f}%')

    # --- 4. 繪製圖表與結果預覽 ---
    model.eval()
    torch.cuda.empty_cache() # 繪圖前最後一次清理
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fig = plt.figure(figsize=(16, 10))
    plt.suptitle("ResNet-50 Optimized Training Results", fontsize=16)

    # 子圖 1：Loss 曲線
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.plot(history['loss'], 'r-o', markersize=4)
    ax1.set_title('Loss Curve')
    ax1.grid(True, alpha=0.3)

    # 子圖 2：Accuracy 曲線
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.plot(history['acc'], 'b-s', markersize=4)
    ax2.set_title('Accuracy Curve (%)')
    ax2.grid(True, alpha=0.3)

    # 預覽測試結果
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    with torch.no_grad():
        test_inputs = images.to(device, memory_format=torch.channels_last)
        outputs = model(test_inputs)
        _, predicted = torch.max(outputs, 1)
        # 計算信心度
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, _ = torch.max(probs, 1)

    for i in range(6):
        ax = fig.add_subplot(2, 4, i + 3)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = np.clip(img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465]), 0, 1)
        ax.imshow(img)
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"P: {classes[predicted[i]]} ({conf[i]:.2f})\nT: {classes[labels[i]]}", color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    torch.save(model.state_dict(), 'resnet50_final.pth')
    print("訓練完成！模型已成功保存並釋放顯存。")