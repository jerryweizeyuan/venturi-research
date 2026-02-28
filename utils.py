import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# utils.py = 训练脚本的"工具箱"
# 包含：
# 1. 数据处理工具
# 2. 可视化工具
# 3. 文件管理工具
def plot_images(images):#将一批图像（batch）显示在一个大画布上，主要用于训练过程中实时查看生成效果
    plt.figure(figsize=(32, 32))## 创建一个大画布（32×32英寸）
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),## 水平拼接所有图像
    ], dim=-2).permute(1, 2, 0).cpu())## 调整维度顺序（C,H,W → H,W,C）
    plt.show()## 显示图像


def save_images(images, path, **kwargs):#将一批图像保存为单个图片文件，主要用于保存训练过程中的中间结果
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):#创建数据加载器，用于批量加载训练数据，是你需要重点修改的函数，因为原始代码是为CIFAR-10图像设计的
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):#创建必要的文件夹结构，确保训练结果有地方保存，避免运行时因文件夹不存在而报错
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
