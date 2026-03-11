import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
#工具函数集合文件

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
    """将一批图像保存为单个图片文件（支持4通道流场数据转3通道RGB）"""#因为标准图片格式最多支持4个通道（RGBA），但您的第4个通道是涡量(ω)，在可视化中通常：要么单独显示，要么用其他方式显示（如颜色映射）
    # 1. 如果输入是4通道，自动取前3个通道用于生成RGB图片
    if images.shape[1] == 4:
        images = images[:, :3, :, :]  # 取 u, v, p 通道
    # 2. 将数据归一化到[0,1]范围，归一化是为了可视化，将流场数据转换为图片格式，不是训练时的归一化（训练时的归一化应该在数据加载时完成），在训练时，通常会对数据进行归一化，但在保存图片时再做一次归一化是为了确保图片能正确显示。
    images = (images - images.min()) / (images.max() - images.min() + 1e-8)
    # 3. 创建网格，转换为标准图片格式并保存
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    print(f"[保存成功] {path}")


def get_data(args):
    from venturi_dataset import VenturiDataset

    dataset = VenturiDataset(#创建一个 VenturiDataset类的实例（对象）
        data_root=args.dataset_path,
        img_size=args.image_size,
        mode='train'
    )

    dataloader = DataLoader(#使用 PyTorch 的 DataLoader类创建一个数据加载器
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    return dataloader


def setup_logging(run_name):
    """创建必要的文件夹结构"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)