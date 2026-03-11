# venturi_dataset.py#加载、预处理和组织您的文丘里管CFD模拟数据，以便输入到扩散模型进行训练。
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np


class VenturiDataset(Dataset):#定义一个名为VenturiDataset的类，它继承自PyTorch的Dataset基类。这意味着这个类必须实现__len__和__getitem__方法。
    """文丘里管流场数据集"""

    def __init__(self, data_root, img_size=64, mode='train'):
        self.data_root = data_root
        self.img_size = img_size
        self.mode = mode
        self.samples = []#初始化一个空列表samples，用于存储所有数据样本的信息。

        # 收集数据
        data_path = os.path.join(data_root, mode)#程序就会去 data/venturi/train/ 文件夹找数据
        if not os.path.exists(data_path):#检查数据路径是否存在。如果不存在，抛出FileNotFoundError异常，并显示错误信息。
            raise FileNotFoundError(f"数据路径不存在: {data_path}")

        # 假设每个子文件夹代表一种几何条件
        for label, geometry in enumerate(sorted(os.listdir(data_path))):
            geometry_path = os.path.join(data_path, geometry)
            if not os.path.isdir(geometry_path):
                continue

            # 收集该几何下的所有流场快照
            files = [f for f in os.listdir(geometry_path)
                     if f.endswith(('.npy', '.npz', '.pt'))]

            for file in sorted(files):
                self.samples.append({
                    'path': os.path.join(geometry_path, file),
                    'label': label,
                    'geometry': geometry
                })

        print(f"加载 {mode} 数据集: {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载数据
        if sample['path'].endswith('.npy'):
            flow = torch.from_numpy(np.load(sample['path'])).float()
        elif sample['path'].endswith('.pt'):
            flow = torch.load(sample['path']).float()
        else:
            raise ValueError(f"不支持的文件格式: {sample['path']}")

        # 调整尺寸
        if flow.dim() == 2:  # [H, W]
            flow = flow.unsqueeze(0)  # [1, H, W]

        if flow.shape[-2:] != (self.img_size, self.img_size):
            flow = F.interpolate(
                flow.unsqueeze(0),  # [1, C, H, W]
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 归一化到[-1, 1]
        flow_min, flow_max = flow.min(), flow.max()
        if flow_max - flow_min > 1e-6:
            flow = 2 * (flow - flow_min) / (flow_max - flow_min) - 1
        else:
            flow = torch.zeros_like(flow)

        return flow, torch.tensor(sample['label'], dtype=torch.long)