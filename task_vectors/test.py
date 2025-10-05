import torch
print(torch.cuda.is_available())  # GPUが使えるかどうか
print(torch.cuda.device_count())  # 使用可能なGPUの数
print(torch.cuda.current_device())  # 現在のデバイス番号
print(torch.cuda.get_device_name(0))  # デバイス名