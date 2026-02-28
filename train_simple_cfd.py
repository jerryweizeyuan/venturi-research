import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cfd_dataset import CFDDataset
from ddpm_conditional import Diffusion
from modules import UNet_conditional

print("🚀 开始CFD扩散模型训练")

# 设置
device = "cpu"  # 先用CPU，稳定
epochs = 3  # 先训练3个epoch看看

# 数据
dataset = CFDDataset(
    flow_dir="E:/CFD_Data/toy_cylinder/flow_fields",
    geom_path="E:/CFD_Data/toy_cylinder/geometry_masks/cylinder.pt",
    num_snapshots=8
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 模型
model = UNet_conditional(c_in=4, c_out=4, num_classes=1, device=device).to(device)
diffusion = Diffusion(img_size=64, device=device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

print(f"📊 训练信息:")
print(f"  设备: {device}")
print(f"  数据量: {len(dataset)} 个样本")
print(f"  Epochs: {epochs}")
print(f"  批次大小: 2")

# 训练
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (flows, labels) in enumerate(dataloader):
        flows = flows.to(device)
        labels = labels.to(device)

        # 扩散过程
        t = diffusion.sample_timesteps(flows.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(flows, t)

        # 预测噪声
        predicted_noise = model(x_t, t, labels)
        loss = nn.MSELoss()(noise, predicted_noise)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 2 == 0:
            print(f"  Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch + 1} 完成，平均损失: {avg_loss:.6f}")

print(f"\n🎉 训练完成！")

# 测试生成
print("\n🧪 测试生成功能...")
model.eval()
labels = torch.tensor([0]).to(device)
generated = diffusion.sample(model, 1, labels)
print(f"生成样本形状: {generated.shape}")
print(f"值范围: [{generated.min():.3f}, {generated.max():.3f}]")

# 保存模型
torch.save(model.state_dict(), "first_cfd_model.pth")
print("💾 模型已保存为 first_cfd_model.pth")