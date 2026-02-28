import torch
import time
from ddpm_conditional import Diffusion
from modules import UNet_conditional

print("=== 真实GPU速度测试 ===")

# 强制使用GPU
device = "cuda"
print(f"强制使用设备: {device}")

# 检查GPU状态
if torch.cuda.is_available():
    print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
else:
    print("✗ GPU不可用，请检查CUDA安装")
    exit()

# 初始化模型
model = UNet_conditional(num_classes=10).to(device)

# 加载权重（如果存在）
try:
    ckpt = torch.load("./models/DDPM_conditional/ckpt.pt", map_location=device)
    model.load_state_dict(ckpt)
    print("✓ 加载预训练权重成功")
except Exception as e:
    print(f"⚠ 无法加载权重: {e}")
    print("  使用随机初始化模型")

# 创建扩散模型（已经修改为100步）
diffusion = Diffusion(noise_steps=100, img_size=64, device=device)
print(f"采样步数: {diffusion.noise_steps}")

# 测试生成速度
print("\n开始测试生成速度...")
labels = torch.tensor([6]).long().to(device)

# 第一次运行（包含编译时间）
print("第1次运行（包含编译时间）...")
start1 = time.time()
with torch.no_grad():
    generated1 = diffusion.sample(model, 1, labels)
end1 = time.time()
print(f"  时间: {end1-start1:.2f}秒")

# 第二次运行（纯生成时间）
print("第2次运行（纯生成时间）...")
start2 = time.time()
with torch.no_grad():
    generated2 = diffusion.sample(model, 1, labels)
end2 = time.time()
print(f"  时间: {end2-start2:.2f}秒")

# 第三次运行（稳定时间）
print("第3次运行（稳定时间）...")
start3 = time.time()
with torch.no_grad():
    generated3 = diffusion.sample(model, 1, labels)
end3 = time.time()
print(f"  时间: {end3-start3:.2f}秒")

print("\n=== 测试结果 ===")
print(f"平均生成时间（后2次）: {(end2-start2 + end3-start3)/2:.2f}秒")
print(f"生成形状: {generated1.shape}")
print(f"值范围: [{generated1.min():.3f}, {generated1.max():.3f}]")

# GPU内存信息
if torch.cuda.is_available():
    print(f"GPU内存使用: {torch.cuda.memory_allocated(0)/1024**3:.3f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
