import numpy as np
from scipy.linalg import blas
import time

def compute_expression_blas(I, K, A, P, R):
    # Step 1: Compute KA = K @ A
    KA = blas.dgemm(alpha=1.0, a=K, b=A)

    # Step 2: Compute I_KA = I - KA
    I_KA = I - KA

    # Step 3: Compute I_KA_P = (I - KA) @ P
    I_KA_P = blas.dgemm(alpha=1.0, a=I_KA, b=P)

    # Step 4: Compute I_KA_P_I_KA_T = I_KA_P @ (I - KA).T
    I_KA_P_I_KA_T = blas.dgemm(alpha=1.0, a=I_KA_P, b=I_KA, trans_b=True)

    # Step 5: Compute KRT = K @ R @ K.T
    KR = blas.dgemm(alpha=1.0, a=K, b=R)
    KRT = blas.dgemm(alpha=1.0, a=KR, b=K, trans_b=True)

    # Step 6: Compute final result
    result = I_KA_P_I_KA_T + KRT

    return result

# Define matrix dimensions
dim = 9300

# Create random matrices
I = np.eye(dim)
K = np.random.rand(dim, dim)
A = np.random.rand(dim, dim)
P = np.random.rand(dim, dim)
R = np.random.rand(dim, dim)

# Compute result
t1 = time.time()
result = compute_expression_blas(I, K, A, P, R)
print(result)
print(time.time() - t1)