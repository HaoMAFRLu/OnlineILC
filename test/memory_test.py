import numpy as np

try:
    dimension = 65 * 550
    print(f"Attempting to create a {dimension} x {dimension} matrix...")
    
    # 尝试分配一个大的float64矩阵
    large_matrix = np.zeros((dimension, dimension), dtype=np.float64)
    
    print("Matrix created successfully.")
    print(f"Matrix dimensions: {large_matrix.shape}")
    print(f"Matrix size: {large_matrix.size}")
    print(f"Matrix total memory usage: {large_matrix.nbytes / (1024 ** 3)} GB")
    
except MemoryError:
    print("MemoryError: Unable to allocate the matrix. Not enough memory.")
except Exception as e:
    print(f"An error occurred: {e}")
