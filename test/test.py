import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, cardinality, stride=1):
        super(Bottleneck, self).__init__()
        D = int(planes / self.expansion)
        self.conv1 = nn.Conv2d(in_planes, D * cardinality, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * cardinality)
        self.conv2 = nn.Conv2d(D * cardinality, D * cardinality, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D * cardinality)
        self.conv3 = nn.Conv2d(D * cardinality, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality, num_classes=10):
        super(ResNeXt, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], cardinality, stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], cardinality, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], cardinality, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], cardinality, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应池化确保输出尺寸为1x1
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, cardinality, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, cardinality, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)  # 使用自适应池化
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNeXt50():
    return ResNeXt(Bottleneck, [3, 4, 6, 3], cardinality=32, num_classes=550)

# 创建 ResNeXt-50 模型实例
model = ResNeXt50()

# 打印模型架构
print(model)

# 示例输入数据
input_tensor = torch.randn(32, 1, 550, 1)  # 32个样本，单通道，高度为550，宽度为1

# 前向传播
output = model(input_tensor)
print(output.shape)  # 输出的形状应该是 (32, num_classes)
