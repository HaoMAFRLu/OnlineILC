import torch
import torch.nn as nn
from torchvision import models

# 加载ResNet模型结构而不加载预训练的参数
resnet18 = models.resnet18(pretrained=False)

# 修改输入层的通道数为1，并调整卷积核的尺寸以适应新的输入尺寸
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)

# 由于输入宽度为1，最大池化层不再适用，因此可以将其删除或调整
resnet18.maxpool = nn.Identity()  # 直接跳过最大池化层

# 假设你希望输出的类别数量是10个类别
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)

# 打印最终修改后的模型结构
print(resnet18)

# 测试新的模型结构是否能够接受指定尺寸的输入
test_input = torch.randn(1, 1, 550, 1)
output = resnet18(test_input)
print(output.shape)  # 应该输出torch.Size([1, 10])
