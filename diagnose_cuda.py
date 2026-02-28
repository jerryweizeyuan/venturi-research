# diagnose_cuda.py
import torch
import sys

print("=== CUDA 诊断报告 ===")
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

if torch.cuda.is_available():
    print(f"\nGPU 设备信息:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("\n✗ CUDA不可用，可能的原因:")
    print("  1. 没有NVIDIA GPU")
    print("  2. 没有安装CUDA驱动")
    print("  3. PyTorch安装的不是CUDA版本")
    print("  4. CUDA和PyTorch版本不匹配")

# 检查conda环境
print(f"\n=== 环境信息 ===")
print(f"conda环境: cfd_ai")