import torch
import sys
sys.path.append('.')

from modules import UNet_conditional
from ddpm_conditional import Diffusion
from torchvision.utils import save_image

print('开始快速采样测试...')
model = UNet_conditional(num_classes=1, device='cpu')
print('模型创建完成')
diffusion = Diffusion(noise_steps=5, img_size=64, device='cpu')
print('扩散模型创建完成')

labels = torch.tensor([0])
print('正在采样（5步）...')
sampled = diffusion.sample(model, n=1, labels=labels, progress=True)

save_image((sampled + 1) / 2, 'test_sample.jpg')
print('图片已保存到: test_sample.jpg')
print(f'图片形状: {sampled.shape}')
