import torch

print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA is available. Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
