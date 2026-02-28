import torch
from ddpm_conditional import Diffusion
from modules import UNet_conditional

print("测试快速生成...")

device = "cpu"
model = UNet_conditional(c_in=4, c_out=4, num_classes=1, device=device)

# 使用修改后的扩散模型（100步）
diffusion = Diffusion(noise_steps=100, img_size=64, device=device)

print(f"采样步数: {diffusion.noise_steps}")

# 测试生成速度
import time
labels = torch.tensor([0])

start_time = time.time()
generated = diffusion.sample(model, 1, labels)
end_time = time.time()

print(f"生成时间: {end_time - start_time:.1f}秒")
print(f"生成形状: {generated.shape}")
print(f"值范围: [{generated.min():.3f}, {generated.max():.3f}]")