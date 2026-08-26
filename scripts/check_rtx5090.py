import sys
import torch

print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('PyTorch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA不可用。请检查NVIDIA驱动和CUDA 12.8+ PyTorch安装。')

name = torch.cuda.get_device_name(0)
prop = torch.cuda.get_device_properties(0)
cc = (prop.major, prop.minor)
print('GPU:', name)
print('Compute capability:', f'{cc[0]}.{cc[1]}')
print('VRAM GiB:', round(prop.total_memory / 1024**3, 2))
print('BF16 supported:', torch.cuda.is_bf16_supported())

if '5090' not in name:
    print('WARNING: 当前GPU不是RTX 5090，但代码仍可在其他CUDA GPU上运行。')
if cc < (12, 0):
    print('WARNING: 该GPU不是Blackwell SM 12.x。')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('ERROR: 当前CUDA/PyTorch环境没有BF16支持。')

x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
y = x @ x
print('BF16 CUDA smoke test:', tuple(y.shape), y.dtype)
print('RTX 5090 environment check: PASS')
