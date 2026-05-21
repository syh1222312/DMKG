import torch
print("PyTorch 版本:", torch.__version__)
print("PyTorch 编译时使用的 CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())
print("可用 GPU 数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))