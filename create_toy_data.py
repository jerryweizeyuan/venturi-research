import torch
import numpy as np
import os

# 1. 创建保存数据的文件夹
data_dir = "E:/CFD_Data/toy_cylinder"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(f"{data_dir}/flow_fields", exist_ok=True)
os.makedirs(f"{data_dir}/geometry_masks", exist_ok=True)

# 2. 创建几何掩码（一个简单的圆形障碍物）
height, width = 128, 128
geom_mask = torch.zeros((1, height, width))  # [1, 128, 128]

# 在中心画一个圆（半径15像素）
for i in range(height):
    for j in range(width):
        if (i - 64) ** 2 + (j - 64) ** 2 <= 225:  # 半径15的圆
            geom_mask[0, i, j] = 1.0

# 保存几何掩码
torch.save(geom_mask, f"{data_dir}/geometry_masks/cylinder.pt")
print(f"几何掩码已保存，形状: {geom_mask.shape}")

# 3. 创建假的流场数据（模拟圆柱绕流）
# 我们创建8个“快照”，模拟一个周期内的变化
num_snapshots = 8
for snap in range(num_snapshots):
    # 创建一个假的流场 [4, 128, 128] = [u, v, p, ω]
    flow_field = torch.zeros((4, height, width))

    # 模拟基本流场：左边来流，右边尾迹
    for i in range(height):
        for j in range(width):
            # 如果不在圆柱内
            if not (i - 64) ** 2 + (j - 64) ** 2 <= 225:
                # u分量：主流方向，在圆柱后减速
                flow_field[0, i, j] = 1.0 - 0.3 * np.exp(-((j - 70) / 30) ** 2)

                # v分量：上下摆动，模拟涡街
                phase = 2 * np.pi * snap / num_snapshots
                flow_field[1, i, j] = 0.2 * np.sin(phase) * np.exp(-((j - 70) / 40) ** 2)

                # 压力：圆柱前高压，后低压
                flow_field[2, i, j] = 0.5 * (1 - (j - 64) / 64) - 0.3 * np.cos(phase)

                # 涡量：圆柱后正负交替
                flow_field[3, i, j] = 0.4 * np.sin(phase + (i - 64) / 30) * np.exp(-((j - 70) / 35) ** 2)

    # 添加一些随机噪声，更真实
    flow_field += 0.05 * torch.randn_like(flow_field)

    # 保存这个快照
    torch.save(flow_field, f"{data_dir}/flow_fields/snapshot_{snap:04d}.pt")
    print(f"快照 {snap} 已保存，形状: {flow_field.shape}")

print(f"\n玩具数据已生成到: {data_dir}")
print(f"包含: 1个几何掩码 + {num_snapshots}个流场快照")