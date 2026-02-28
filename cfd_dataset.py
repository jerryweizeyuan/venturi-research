import torch
import torch.nn.functional as F  # 添加这行
from torch.utils.data import Dataset, DataLoader
import os


class CFDDataset(Dataset):
    def __init__(self, flow_dir, geom_path, num_snapshots=8):
        self.flow_dir = flow_dir
        # 加载几何掩码并下采样到64x64
        geom = torch.load(geom_path)  # [1, 128, 128]
        self.geom = F.interpolate(geom.unsqueeze(0), size=64, mode='nearest').squeeze(0)  # 下采样
        self.num_snapshots = num_snapshots

    def __len__(self):
        return self.num_snapshots

    def __getitem__(self, idx):
        # 加载流场快照并下采样到64x64
        flow_path = os.path.join(self.flow_dir, f"snapshot_{idx:04d}.pt")
        flow = torch.load(flow_path)  # [4, 128, 128]
        flow = F.interpolate(flow.unsqueeze(0), size=64, mode='bilinear').squeeze(0)  # 下采样

        # 归一化
        flow = (flow - flow.mean()) / (flow.std() + 1e-8)

        # 返回流场和数字标签0（代表圆柱）
        return flow, torch.tensor(0, dtype=torch.long)  # 标签为0


# 测试数据集（可以保留这部分）
if __name__ == "__main__":
    dataset = CFDDataset(
        flow_dir="E:/CFD_Data/toy_cylinder/flow_fields",
        geom_path="E:/CFD_Data/toy_cylinder/geometry_masks/cylinder.pt",
        num_snapshots=8
    )

    print(f"数据集大小: {len(dataset)}")

    # 获取一个样本
    flow, label = dataset[0]
    print(f"流场形状: {flow.shape}")
    print(f"标签: {label}")

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_idx, (flows, labels) in enumerate(dataloader):
        print(f"批次 {batch_idx}: flows形状={flows.shape}, labels={labels}")
        if batch_idx == 0:
            break