import torch
import torch.nn as nn

# 定义一个简单的神经网络
class MyNetwork(nn.Module):
    def __init__(self, in_channel, output_dim):
        super(MyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128 * in_channel * 17, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, output_dim, bias=True),
        )

    def forward(self, x):
        return self.fc(x)

# 实例化模型
in_channel = 3  # 示例输入通道数
output_dim = 10  # 示例输出维度
model = MyNetwork(in_channel, output_dim)

# 准备输入数据
input_data = torch.randn(1, 128 * in_channel * 17)  # 示例输入数据

# 定义一个变量来存储中间层的输出
intermediate_outputs = {}

# 定义钩子函数
def hook(module, input, output):
    intermediate_outputs['layer_output'] = output

# 注册钩子到特定的层（例如第二个线性层）
hook_handle = model.fc[2].register_forward_hook(hook)

# 执行前向传播
output = model(input_data)

# 取消注册钩子（可选）
hook_handle.remove()

# 打印中间层的输出
print("Intermediate output from the second linear layer:")
print(intermediate_outputs['layer_output'])
