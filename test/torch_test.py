import torch
import time

# 定义矩阵维度
dim = 9300

# 创建随机矩阵并将其移动到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

I = torch.eye(dim, device=device)
K = torch.rand(dim, dim, device=device)
A = torch.rand(dim, dim, device=device)
P = torch.rand(dim, dim, device=device)
R = torch.rand(dim, dim, device=device)

# 预热 GPU
torch.matmul(K, A)

# 测量执行时间
start_time = time.time()

# 计算 KA = K @ A
KA = torch.matmul(K, A)

# 计算 I_KA = I - KA
I_KA = I - KA

# 计算 I_KA_P = I_KA @ P
I_KA_P = torch.matmul(I_KA, P)

# 计算 I_KA_P_I_KA_T = I_KA_P @ I_KA.T
I_KA_P_I_KA_T = torch.matmul(I_KA_P, I_KA.t())

# 计算 KRT = K @ R @ K.T
KRT = torch.matmul(torch.matmul(K, R), K.t())

# 计算最终结果
result = I_KA_P_I_KA_T + KRT

# 结束计时并输出时间
elapsed_time = time.time() - start_time
print(f"Elapsed time on GPU: {elapsed_time} seconds")
