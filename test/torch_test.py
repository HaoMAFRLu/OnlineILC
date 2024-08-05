import torch
import time

dim = 9350

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

I = torch.eye(dim, device=device)
K = torch.rand(dim, dim, device=device)
A = torch.rand(dim, dim, device=device)
P = torch.rand(dim, dim, device=device)
R = torch.rand(dim, dim, device=device)

torch.matmul(K, A)

start_time = time.time()

KA = torch.matmul(K, A)

I_KA = I - KA

I_KA_P = torch.matmul(I_KA, P)

I_KA_P_I_KA_T = torch.matmul(I_KA_P, I_KA.t())

KRT = torch.matmul(torch.matmul(K, R), K.t())

result = I_KA_P_I_KA_T + KRT

elapsed_time = time.time() - start_time
print(f"Elapsed time on GPU: {elapsed_time} seconds")
