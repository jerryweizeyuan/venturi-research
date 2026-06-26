# ddpm_scheduler.py
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm


class DDPMScheduler:
    """DDPM扩散调度器"""

    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device

        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        # 预计算扩散参数
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # 反向过程参数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # 后验方差参数
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def add_noise(self, original, noise, timesteps):
        """前向扩散：添加噪声到图像"""
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[timesteps][:, None, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[timesteps][:, None, None, None]

        noisy = sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise
        return noisy

    def step(self, noise_pred, timesteps, x_t):
        """反向扩散：去噪一步"""
        # 确保时间步是张量
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor([timesteps], device=self.device)

        # 获取当前时间步的参数
        timesteps = timesteps.to(self.device)
        alphas_t = self.alphas[timesteps][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[timesteps][:, None, None, None]

        # 计算预测的原始图像
        pred_original = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / self.sqrt_alpha_cumprod[timesteps][
            :, None, None, None]

        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
                    x_t - self.betas[timesteps][:, None, None, None] * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        # 计算方差
        if timesteps[0] > 0:
            posterior_variance_t = self.posterior_variance[timesteps][:, None, None, None]
            noise = torch.randn_like(x_t)
        else:# 当t=0时，后验方差为
            posterior_variance_t = torch.zeros_like(x_t)  # 改为张量形式
            noise = torch.zeros_like(x_t)

        # 采样
        x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise

        return x_prev

    def sample_timesteps(self, batch_size):
        """采样时间步"""
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def noise_images(self, x, t):
        """为图像添加噪声（用于训练）"""
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        noise = torch.randn_like(x)
        noisy_images = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise

        return noisy_images, noise

    def sample(self, model, n, geometry_masks, cfg_scale=3.0):
        """从噪声开始生成样本"""
        model.eval()

        with torch.no_grad():
            # 从随机噪声开始
            x = torch.randn((n, 4, self.img_size, self.img_size), device=self.device)

            # 确保几何掩码形状正确
            if geometry_masks.shape[0] == 1 and n > 1:
                geometry_masks = geometry_masks.repeat(n, 1, 1, 1)

            # 反向扩散过程
            for i in tqdm(range(self.timesteps - 1, -1, -1), desc="采样"):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)

                # 预测噪声
                noise_pred = model(x, t, geometry_masks)

                # 分类器自由引导
                if cfg_scale > 0:
                    uncond_noise = model(x, t, torch.zeros_like(geometry_masks))
                    noise_pred = uncond_noise + cfg_scale * (noise_pred - uncond_noise)

                # 去噪一步
                x = self.step(noise_pred, t, x)

            return x

    def q_sample(self, x_0, t, noise=None):
        """前向扩散采样 q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise


if __name__ == "__main__":
    # 测试调度器
    scheduler = DDPMScheduler(timesteps=200, device="cpu")

    # 测试添加噪声
    batch_size = 2
    img_size = 64
    x = torch.randn(batch_size, 4, img_size, img_size)
    t = torch.randint(0, 200, (batch_size,))

    noisy_x, noise = scheduler.noise_images(x, t)
    print(f"✅ 噪声添加测试通过")
    print(f"原始形状: {x.shape}")
    print(f"噪声形状: {noise.shape}")
    print(f"加噪后形状: {noisy_x.shape}")

    # 测试反向步骤
    t_single = torch.tensor([100])
    x_prev = scheduler.step(noise, t_single, noisy_x)
    print(f"✅ 反向步骤测试通过")
    print(f"去噪后形状: {x_prev.shape}")
