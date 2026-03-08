import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):  #noise_steps=100：扩散总步数
        self.noise_steps = noise_steps #计算并存储
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)  #如果beta是加噪比例，alpha就是保留原图的比例，调用prepare_noise_schedule方法生成beta序列
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  #计算累积乘积，从开始到当前步总共保留的原图比例

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)  #生成从beta_start到beta_end的等差数列

    def noise_images(self, x, t):  #x：输入参数，清晰的原始图片；t：输入参数，时间步（第几步加噪声），这部分代码固定
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]  #从alpha_hat序列中取出第t个值，t通常是数组，返回对应长度的数组，其形状为 [14, 通道数, 高度, 宽度]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]  #扩展维度，便于广播计算
        Ɛ = torch.randn_like(x)  #生成与x同形状的随机数
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ  #加噪图片 = (保留系数 × 原图) + (噪声系数 × 随机噪声)
#前向扩散
    def sample_timesteps(self, n):  #生成随机时间步，表示每个样本应该加噪到第几步
        return torch.randint(low=1, high=self.noise_steps, size=(n,))  #会生成一个包含 n个随机整数的张量（数组），每个整数都在 [1, self.noise_steps)的范围内。这意味着：批次中的第N个样本将被加噪到第x步。

    # 反向生成
    def sample(self, model, n, labels, cfg_scale=3):  #n：要生成的样本数量，labels：条件标签，控制生成内容，cfg_scale=3：分类器自由引导的强度，可以自己改，越大条件约束更强
        logging.info(f"Sampling {n} new images....")  #在控制台输出信息，提示开始生成过程
        model.eval()  #训练时用model.train()，测试/预测时用model.eval()
        with torch.no_grad():  #大幅减少内存使用，加速计算
            x = torch.randn((n, 4, self.img_size, self.img_size)).to(self.device)  #n：生成n个样本,4：4个通道（u,v,p,ω物理量）-生成标准正态分布随机数，这个张量可以看作是n张图像，每张图像有4个通道，每个通道是self.img_size x self.img_size的像素网格。初始化x维纯噪声图片（随机生成）
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):  #reversed(range(1, 100))：从99,98,...,1倒序循环，position=0：进度条位置
                t = (torch.ones(n) * i).long().to(self.device)  #创建长度为n的全1张量，创建时间步张量，告诉模型当前是第几步
                predicted_noise = model(x, t, labels)  #让U-Net预测噪声，根据条件标签控制生成内容，x是当前状态，有条件预测
                if cfg_scale > 0:  #当 cfg_scale > 0 时启用 CFG
                    uncond_predicted_noise = model(x, t, None)  #不看标签，通用去噪
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)  #线性插值：在两者间平衡
                alpha = self.alpha[t][:, None, None, None]   # 单步保留比例，按需读取
                alpha_hat = self.alpha_hat[t][:, None, None, None] # 累积保留比例
                beta = self.beta[t][:, None, None, None] # 噪声强度
                if i > 1:# 前99步添加随机噪声
                    noise = torch.randn_like(x) # 生成随机噪声
                else: # 最后一步不加随机噪声
                    noise = torch.zeros_like(x) # 零噪声（确定性）增加生成多样性（前99步）确保收敛（最后一步确定）
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise#x是一批由模型生成的、清晰的、符合指定条件标签的图片。
        model.train()#sample函数的最后一步，用于将模型状态恢复为训练模式
        #x = (x.clamp(-1, 1) + 1) / 2
        #x = (x * 255).type(torch.uint8)
        return x
def train(args):#定义主训练函数，args包含所有训练配置参数
    setup_logging(args.run_name)#调用工具函数设置日志系统，创建日志文件，记录训练过程，args.run_name是实验名称，用于区分不同训练任务
    device = args.device#获取计算设备
    dataloader = get_data(args)#从 utils.py文件导入的函数，创建PyTorch的 DataLoader对象，用于批量加载训练数据，需要修改此函数以加载文丘里管流场数据
    model = UNet_conditional(num_classes=args.num_classes).to(device)#创建条件U-Net模型实例，num_classes：条件标签的类别数，`num_classes`必须与您数据集中条件标签的总类别数一致。例如，您有5种文丘里管几何，这里就应设为5。
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)#使用AdamW优化器（Adam with weight decay），负责根据损失（预测误差）更新模型参数。学习率：lr=args.lr（默认3e-4）
    mse = nn.MSELoss()#定义损失函数。模型的任务是“预测噪声”，损失即为预测噪声与真实噪声的均方误差，用于比较预测噪声和真实噪声，公式：MSE = 1/n * Σ(预测值 - 真实值)²
    diffusion = Diffusion(img_size=args.image_size, device=device)#创建扩散模型类，包含加噪、去噪、时间步采样等方法
    logger = SummaryWriter(os.path.join("runs", args.run_name))#创建TensorBoard日志记录器，用于可视化训练过程中的损失曲线，查看命令：tensorboard --logdir=runs
    l = len(dataloader)#计算每个epoch的批次数量，用于计算全局训练步数
    #在训练过程中，维护一个模型参数的“滑动平均”（ema_model），而非直接使用当前模型（model）。这通常能使生成结果更稳定、质量更高。您最终应使用ema_model进行生成，因为它通常更好。代码中也会同时保存这两个模型
    ema = EMA(0.995)#指数移动平均（Exponential Moving Average）计算器，参数0.995是平滑系数，用于平滑模型参数更新，提高稳定性
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)#创建原模型的深拷贝作为EMA模型，.eval()：设置为评估模式，.requires_grad_(False)：关闭梯度计算，节省内存，EMA模型用于生成更稳定的结果

    for epoch in range(args.epochs):#遍历所有训练轮次，range(args.epochs)生成从0到args.epochs-1的整数序列，默认训练300个epoch，epoch是一个循环变量，由 for epoch in range(args.epochs):这行循环语句自动定义并赋值。
        logging.info(f"Starting epoch {epoch}:")#使用Python的f-string格式化字符串，在控制台输出当前epoch开始信息
        pbar = tqdm(dataloader)#用tqdm包装数据加载器，显示训练进度、处理速度、剩余时间，tqdm是Python进度条库
        for i, (images, labels) in enumerate(pbar):#遍历当前epoch的所有批次，enumerate()：同时获取索引i和数据，每次循环得到批次索引 i、图片数据 images、标签数据 labels
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)#sample_timesteps(n)：生成n个随机时间步，images.shape[0]：内置的 shape 属性，表示批次大小，# 为这批数据中的每张图片，随机分配一个“加噪步数”t（范围1-99）。这代表每张图片被破坏的程度不同
            x_t, noise = diffusion.noise_images(images, t)## 根据公式 `x_t = √ᾱ_t * 原图 + √(1-ᾱ_t) * 随机噪声` 对清晰图片`images`加噪。
            if np.random.random() < 0.1:#np.random.random()：NumPy函数，返回[0,1)之间的随机浮点数
                labels = None
            predicted_noise = model(x_t, t, labels)#调用U-Net模型，模型认为应该去除的噪声模式
            loss = mse(noise, predicted_noise)#mse()：均方误差损失函数，比较两个张量的差异

            optimizer.zero_grad()#清空上一批次计算的梯度，PyTorch默认累积梯度，必须清零重新计算
            loss.backward()#自动计算损失对所有参数的梯度
            optimizer.step()#根据梯度更新模型参数
            ema.step_ema(ema_model, model)#更新指数移动平均模型

            pbar.set_postfix(MSE=loss.item())#pbar：tqdm进度条对象，.set_postfix()：在进度条后显示附加信息，loss.item()：获取标量损失值
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)# TensorBoard记录
#定期保存和验证
      #  if epoch % 10 == 0:#%：取模运算符，每10个epoch执行一次，第0,10,20,...,290,300轮执行，这里的 10 表示验证和保存检查点的频率，即每10个epoch执行一次验证和保存。这个数字不需要修改
         #   labels = torch.arange(10).long().to(device)#orch.arange(10)：创建0-9的序列，结果：tensor([0,1,2,3,4,5,6,7,8,9])， # 将这里的 10 改为与 args.num_classes（在launch()函数中） 相同的值
            # 确保标签不超出范围
         #   labels = torch.clamp(labels, 0, args.num_classes - 1)#这行代码是测试时临时加的，正式开始时要删除这行代码
          #  sampled_images = diffusion.sample(model, n=len(labels), labels=labels)#保证 len(labels) == n，目的是一次性为数据集中的每一个类别（由标签代表）生成一个对应的样本，
         #   ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
           # plot_images(sampled_images)  # 调用工具函数显示图片
           # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))#save_images()：保存图片为文件，跨平台路径拼接，f-string格式化文件名
           # save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))#图片保存在results下的实验名称（也就是args.run_name）下
           # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))#将当前训练中的原始U-Net模型的所有可学习参数（权重、偏置等）保存到一个文件中，# 路径示例：models/DDPM_conditional/ckpt.pt
           # torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))#将EMA模型的参数保存到另一个独立的文件中。如前所述，这个模型通常是生成质量更高、更稳定的版本。
           # torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))#保存优化器的当前状态。这包括优化器内部为每个参数维护的动量（momentum）、二阶矩估计等动态信息。
        if epoch % 10 == 0:
              labels = torch.arange(10).long().to(device)
              labels = torch.clamp(labels, 0, args.num_classes - 1)

              sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
              ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

            # === 核心：先保存完整4通道原始数据 ===
              torch.save(sampled_images, os.path.join("results", args.run_name, f"{epoch}_raw.pt"))
              torch.save(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema_raw.pt"))

           # === 再保存3通道可视化图片 ===
              save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
              save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))

    # 保存模型（此部分不变）
              torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
              torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
              torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

    # 可选：如果需要看图，可以临时取消下一行的注释，但程序会暂停直到您关闭图片窗口
    # plot_images(sampled_images[:, :3, :, :])

def launch():#定义一个名为 launch的函数，作为整个训练过程的启动入口。
    import argparse#导入Python的argparse模块。这个模块用于解析命令行参数，虽然在这个函数中并未实际从命令行获取参数，但通常用于构建可配置的训练脚本
    parser = argparse.ArgumentParser()#创建一个参数解析器对象。这个对象可以定义和解析命令行参数，方便用户通过命令行修改训练配置。
    args = parser.parse_args()#解析命令行参数。由于没有预先定义任何参数，这里只是返回一个空的命名空间对象。通常我们会先调用parser.add_argument()定义参数，再解析。
    args.run_name = "DDPM_conditional"#设置训练运行名称。这个名称用于：创建保存模型、日志、结果的子目录，在TensorBoard中标识本次实验
    args.epochs = 1#设置训练的总轮数（epochs）。一个epoch表示模型遍历完整训练集一次。300表示模型将看300遍整个训练集。
    args.batch_size = 1#设置批次大小。每次训练迭代中使用的样本数量。14表示每次更新模型参数时，使用14张图片计算梯度
    args.image_size = 64#设置输入图像的尺寸。64表示图片将被处理为64×64像素。注意：CIFAR-10原始是32×32，这里可能使用了预处理的上采样版本。
    args.num_classes = 1#设置类别数量。CIFAR-10数据集共有10个类别
    args.dataset_path = "data/venturi"#设置训练数据集的路径
    args.device = "cpu"#设置计算设备。"cuda"表示使用NVIDIA GPU进行加速计算。如果没有可用的GPU，应该改为"cpu"。
    args.lr = 3e-4#设置学习率。3e-4（即0.0003）是Adam优化器常用的学习率值，控制每次参数更新的步长
    train(args)#调用之前定义的train()函数，传入所有配置参数，开始训练过程。


if __name__ == '__main__':#Python程序的入口点检查。只有当脚本被直接运行时（而不是被其他模块导入），才会执行下面的代码块。
    launch()#调用launch()函数，启动整个训练流程。
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

