import torch
import sys
sys.path.append('.')

from modules import UNet_conditional
from ddpm_conditional import Diffusion
from torchvision.utils import save_image

print('快速采样测试（修正版）')

# 创建模型
model = UNet_conditional(num_classes=1, device='cpu')
print('模型创建完成')

# 创建扩散模型，只使用2步
diffusion = Diffusion(noise_steps=2, img_size=64, device='cpu')
print('扩散模型创建完成')

# 采样 - 根据实际参数调整
labels = torch.tensor([0])
print('开始采样...')

try:
    # 先尝试带标签
    sampled = diffusion.sample(model, n=1, labels=labels)
    print('采样成功（带标签）')
except TypeError as e:
    print(f'带标签失败: {e}')
    # 尝试不带标签
    sampled = diffusion.sample(model, n=1)
    print('采样成功（不带标签）')

# 保存
save_image(sampled, 'test_output.jpg')
print(f'图片已保存: test_output.jpg, 形状: {sampled.shape}')
