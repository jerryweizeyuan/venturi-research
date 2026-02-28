import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cfd_dataset import CFDDataset
from ddpm_conditional import Diffusion
from modules import UNet_conditional

print("=" * 50)
print("开始CFD扩散模型训练测试")
print("=" * 50)

# 1. 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 2. 加载数据
print("\n[1/4] 加载数据...")
dataset = CFDDataset(
    flow_dir="E:/CFD_Data/toy_cylinder/flow_fields",
    geom_path="E:/CFD_Data/toy_cylinder/geometry_masks/cylinder.pt",
    num_snapshots=8
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print(f"   数据集大小: {len(dataset)} 个样本")
print(f"   数据加载器: {len(dataloader)} 个批次")

# 3. 创建模型
print("\n[2/4] 创建模型...")
model = UNet_conditional(c_in=4, c_out=4, num_classes=1, device=device).to(device)
diffusion = Diffusion(img_size=64, device=device)  # 注意：改为64
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

print(f"   模型输入通道: 4 (u, v, p, ω)")
print(f"   模型输出通道: 4")
print(f"   扩散模型尺寸: 64x64")
print(f"   学习率: {1e-4}")

# 4. 测试训练一个批次
print("\n[3/4] 测试单个训练批次...")
model.train()

for batch_idx, (flows, labels) in enumerate(dataloader):
    if batch_idx >= 1:  # 只测试第一个批次
        break

    flows = flows.to(device)
    labels = labels.to(device)

    print(f"   批次 {batch_idx}:")
    print(f"     - 流场形状: {flows.shape}")
    print(f"     - 标签: {labels}")

    # 扩散过程
    t = diffusion.sample_timesteps(flows.shape[0]).to(device)
    print(f"     - 时间步: {t}")

    x_t, noise = diffusion.noise_images(flows, t)
    print(f"     - 加噪后形状: {x_t.shape}")
    print(f"     - 噪声形状: {noise.shape}")

    # 预测噪声
    predicted_noise = model(x_t, t, labels)
    print(f"     - 预测噪声形状: {predicted_noise.shape}")

    # 计算损失
    loss = nn.MSELoss()(noise, predicted_noise)
    print(f"     - 初始损失: {loss.item():.6f}")

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"     - 梯度更新完成")

# 5. 测试生成
print("\n[4/4] 测试生成样本...")
model.eval()
with torch.no_grad():
    n = 1  # 生成1个样本
    labels = torch.tensor([0] * n).to(device)  # 标签0代表圆柱

    print(f"   生成 {n} 个样本，标签: {labels}")
    generated = diffusion.sample(model, n, labels, cfg_scale=0)
    print(f"   生成样本形状: {generated.shape}")

    if generated.shape == torch.Size([n, 3, 64, 64]):
        print(f"   ⚠️ 注意: 输出是3通道，但CFD需要4通道")
        print(f"   需要进一步修改生成部分")
    else:
        print(f"   生成形状: {generated.shape}")

print("\n" + "=" * 50)
print("训练测试完成！")
print("=" * 50)
print("\n📋 结果总结:")
print("✅ 数据加载成功")
print("✅ 模型创建成功")
print("✅ 单个训练批次完成")
print("✅ 梯度更新正常")
print("\n🎯 下一步:")
print("1. 运行完整训练（多个epoch）")
print("2. 修改生成部分输出4通道")
print("3. 可视化训练结果")