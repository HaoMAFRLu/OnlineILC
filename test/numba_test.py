import numpy as np
import numba as nb
import time

@nb.jit(nopython=True)
def dot_product(A, B):
    return np.dot(A, B)

@nb.jit(nopython=True)
def update_P(I, K, A, P, R):
    KA = dot_product(K, A)
    I_KA = I - KA
    KRT = dot_product(dot_product(K, R), K.T)
    result = dot_product(dot_product(I_KA, P), I_KA.T) + KRT
    return result

dim = 9000

I = np.eye(dim, dtype=np.float32)
K = np.random.rand(dim, dim).astype(np.float32)
A = np.random.rand(dim, dim).astype(np.float32)
P = np.random.rand(dim, dim).astype(np.float32)
R = np.random.rand(dim, dim).astype(np.float32)

t1 = time.time()
update_P(I, K, A, P, R)
print(time.time() - t1)

print('start')

num_trials = 3
total_time = 0.0
for _ in range(num_trials):
    start_time = time.time()
    result = update_P(I, K, A, P, R)
    end_time = time.time()
    print(end_time - start_time)
    total_time += end_time - start_time

average_time = total_time / num_trials
print(f"Average Numba JIT execution time: {average_time} seconds")
