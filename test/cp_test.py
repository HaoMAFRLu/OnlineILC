import cupy as cp

# 定义矩阵维度
dim = 9300

# 创建随机矩阵
I = cp.eye(dim)
K = cp.random.rand(dim, dim)
A = cp.random.rand(dim, dim)
P = cp.random.rand(dim, dim)
R = cp.random.rand(dim, dim)

# 计算表达式
KA = cp.dot(K, A)
I_KA = I - KA
KRT = cp.dot(cp.dot(K, R), K.T)
result = cp.dot(cp.dot(I_KA, P), I_KA.T) + KRT

print(result)
