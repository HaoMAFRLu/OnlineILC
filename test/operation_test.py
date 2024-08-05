import numpy as np

# 定义矩阵 A 和向量 v
A = np.random.randn(50, 50)
v = np.random.randn(17)
I = np.zeros((50-17, 17))
# 定义对角矩阵 S 的对角元素 d
d = np.random.randn(17)

# 构造对角矩阵 S
S = np.diag(d)
K = np.vstack((S, I))
# 计算 G 矩阵
G = A @ np.vstack((np.diag(v), I))

# 计算 A * S * v
ASv = A @ (K @ v)

# 计算 G * d
Gd = G @ d

print("A * S * v:")
print(ASv)

print("\nG * d:")
print(Gd)

# 验证是否相等
print("\n是否相等:", np.allclose(ASv, Gd))
