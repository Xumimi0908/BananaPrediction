import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os

# 数据预处理（适配MobileNetV3）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_datasets():
    train_dataset = datasets.ImageFolder(root='./mydata/trainData', transform=transform)
    test_dataset = datasets.ImageFolder(root='./mydata/testData', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, train_dataset.classes

class BananaMobileNetV3(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)
        # 修改分类层
        self.mobilenet.classifier[3] = nn.Linear(
            self.mobilenet.classifier[3].in_features, num_classes
        )
        self.transform = transform

    def forward(self, x):
        return self.mobilenet(x)

    def train_model(self, train_loader, epochs=10):
        """训练模型"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.train()  # 设置为训练模式
        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                # 前向传播 + 计算损失
                outputs = self(images)
                loss = criterion(outputs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # 保存模型
        torch.save(self.state_dict(), "mobile_model.pth")
        print("模型训练完成并已保存!")