import torch

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available.")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU is not available.")

if __name__ == "__main__":
    check_gpu()