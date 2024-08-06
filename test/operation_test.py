import numpy as np

# 定义矩阵 A 和向量 v
A = np.random.randn(550, 550)
v = np.random.randn(551)

# 定义对角矩阵 S 的对角元素 d
d = np.random.randn(550)

# 构造 S 矩阵
S = np.zeros((550, 551))
np.fill_diagonal(S, d)

# 计算 A * S * v
ASv = A @ (S @ v)

# 构造 G 矩阵
G = A @ np.diag(v[:550])

# 计算 G * d
Gd = G @ d

print("A * S * v:")
print(ASv)

print("\nG * d:")
print(Gd)

# 验证是否相等
print("\n是否相等:", np.allclose(ASv, Gd))
