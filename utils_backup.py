import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


# utils.py = 训练脚本的"工具箱"
# 已修改为支持文丘里管流场数据

def plot_images(images):
    """将一批图像（batch）显示在一个大画布上"""
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    """将一批图像保存为单个图片文件"""
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    """
    加载文丘里管流场数据集
    参数:
        args: 包含数据集路径(dataset_path)、图像大小(image_size)、批大小(batch_size)等参数
    返回:
        dataloader: PyTorch数据加载器
    """
    # 导入自定义的文丘里管数据集类
    from venturi_dataset import VenturiDataset

    # 创建训练数据集实例
    dataset = VenturiDataset(
        data_root=args.dataset_path,  # 数据集根目录，例如: "data/venturi"
        img_size=args.image_size,  # 图像尺寸，例如: 64
        mode='train'  # 模式：'train' 或 'test'
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # 批处理大小
        shuffle=True,  # 训练时打乱数据
        num_workers=0,  # Windows上建议设为0
        drop_last=True  # 丢弃最后一个不完整的批次
    )

    return dataloader


def setup_logging(run_name):
    """创建必要的文件夹结构"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)