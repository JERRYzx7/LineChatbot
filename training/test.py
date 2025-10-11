import torch
print(torch.__version__)           # 確認 PyTorch 版本
print(torch.version.cuda)          # 確認 CUDA 版本
print(torch.cuda.is_available())   # 應該會回傳 True
print(torch.cuda.get_device_name(0))

