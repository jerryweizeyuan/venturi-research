import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# 设置保存目录
SAVE_DIR = "cylinder_diffusion_models"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 模型将保存到: {os.path.abspath(SAVE_DIR)}")


# 自适应物理权重函数
def get_physics_weight(epoch, total_epochs, base_weight=0.0005):
    if total_epochs <= 0:
        return 0.0
    progress = epoch / total_epochs
    w_start = 0.0001
    w_end = base_weight
    return w_start + (w_end - w_start) * progress


# 查找数据文件
def find_data_file():
    global DATA_PATH
    possible_paths = [
        "raw_data/cylinder_training_data/cylinder_training_data.npz",
    ]
    print("🔍 查找数据文件...")
    for path in possible_paths:
        if os.path.exists(path):
            DATA_PATH = path
            print(f"✅ 找到数据文件: {path}")
            return path
    print("❌ 找不到数据文件")
    return None


DATA_PATH = find_data_file()
if DATA_PATH is None:
    sys.exit(1)

# 导入必要的模块
try:
    from fixed_conditional_unet import FixedConditionalUNet
    from ddpm_scheduler import DDPMScheduler
    from physics_constraints_corrected import SimplePhysicsConstraints
    print("✅ 成功导入所有模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)


class ConditionalDataset:
    def __init__(self, npz_file):
        print(f"\n📂 加载数据文件: {npz_file}")
        print(f"   文件存在: {os.path.exists(npz_file)}")
        try:
            data = np.load(npz_file, allow_pickle=True)
            self.flow_fields = torch.from_numpy(data['flow_fields'].astype(np.float32))
            self.geometry_masks = torch.from_numpy(data['geometry_masks'].astype(np.float32))
            print(f"\n✅ 数据加载成功")
            print(f"   样本数: {len(self.flow_fields)}")
            print(f"   流场形状: {self.flow_fields.shape}")
            print(f"   几何掩码形状: {self.geometry_masks.shape}")
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            raise

    def __len__(self):
        return len(self.flow_fields)

    def __getitem__(self, idx):
        flow = self.flow_fields[idx].clone()
        mask = self.geometry_masks[idx].clone()
        for c in range(flow.shape[0]):
            channel = flow[c]
            if channel.std() > 1e-6:
                flow[c] = (channel - channel.mean()) / (channel.std() + 1e-6)
        return flow, mask


def train_with_physics(enable_physics=True, physics_weight=0.1, save_every=5, resume_path=None,epochs=200):
    #epochs = 200
    batch_size = 2
    image_size = 64
    noise_steps = 1000
    lr = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("训练带物理约束的条件扩散模型（修正版）")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {lr}")
    print(f"扩散步数: {noise_steps}")
    print(f"启用物理约束: {enable_physics}")
    print(f"物理约束权重: {physics_weight} (从0.001渐进增加到{physics_weight})")
    print(f"数据文件: {DATA_PATH}")
    print(f"模型保存目录: {SAVE_DIR}")
    if resume_path:
        print(f"预训练模型: {resume_path}")
    print("=" * 60)

    # 加载数据
    print("\n📥 加载数据...")
    try:
        dataset = ConditionalDataset(DATA_PATH)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        print(f"✅ 数据加载器创建成功，批次数量: {len(dataloader)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None

    # 创建模型
    print("🤖 创建条件模型...")
    model = FixedConditionalUNet(c_in=4, c_out=4, time_dim=256, device=device).to(device)
    print(f"✅ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载预训练权重
    if resume_path is not None:
        print(f"📂 加载预训练模型: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 预训练权重加载成功")

    # 如果启用物理约束，创建物理约束模块
    if enable_physics:
        print("🔬 创建物理约束模块（简化版）...")
        physics_module = SimplePhysicsConstraints(
            divergence_weight=0.1,
            boundary_weight=1.0,
            device=device
        )
        print(f"✅ 简化版物理约束模块创建成功")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    print("⏰ 创建扩散调度器...")
    diffusion = DDPMScheduler(
        timesteps=noise_steps,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=image_size,
        device=device
    )

    losses = []
    physics_losses = [] if enable_physics else []
    diffusion_losses = []
    physics_weights_log = [] if enable_physics else []

    print("\n🚀 开始训练...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_physics_loss = 0
        epoch_diffusion_loss = 0
        epoch_continuity_loss = 0
        epoch_boundary_loss = 0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (flow, mask) in enumerate(pbar):
            try:
                flow = flow.to(device)
                mask = mask.to(device)

                t = diffusion.sample_timesteps(flow.shape[0])
                x_t, noise = diffusion.noise_images(flow, t)
                predicted_noise = model(x_t, t, mask)
                diffusion_loss = criterion(noise, predicted_noise)

                physics_loss = torch.tensor(0.0, device=device)
                physics_details = {}

                if enable_physics and physics_weight > 0:
                    with torch.no_grad():
                        alpha_bar = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1)
                        estimated_flow = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
                        estimated_velocity = estimated_flow[:, :2, :, :]

                    physics_loss, physics_details = physics_module(
                        velocity_field=estimated_velocity,
                        mask=mask.unsqueeze(1) if mask.dim() == 3 else mask,
                        epoch=epoch,
                        total_epochs=epochs
                    )
                    current_physics_weight = get_physics_weight(epoch, epochs, physics_weight)
                else:
                    current_physics_weight = 0.0

                if enable_physics:
                    total_loss = diffusion_loss + current_physics_weight * physics_loss
                else:
                    total_loss = diffusion_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_diffusion_loss += diffusion_loss.item()
                epoch_physics_loss += physics_loss.item() if enable_physics else 0

                if enable_physics and 'continuity' in physics_details:
                    epoch_continuity_loss += physics_details.get('continuity', 0)
                if enable_physics and 'boundary' in physics_details:
                    epoch_boundary_loss += physics_details.get('boundary', 0)

                batch_count += 1

                if enable_physics:
                    pbar.set_postfix({
                        'total': f"{total_loss.item():.4f}",
                        'diff': f"{diffusion_loss.item():.4f}",
                        'phy': f"{physics_loss.item():.4f}",
                        'w_phy': f"{current_physics_weight:.4f}"
                    })
                else:
                    pbar.set_postfix(loss=f"{total_loss.item():.4f}")

            except Exception as e:
                print(f"\n❌ 批次 {batch_idx} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_diff_loss = epoch_diffusion_loss / batch_count
            avg_phy_loss = epoch_physics_loss / batch_count if enable_physics else 0
            avg_cont_loss = epoch_continuity_loss / batch_count if enable_physics else 0
            avg_bound_loss = epoch_boundary_loss / batch_count if enable_physics else 0

            losses.append(avg_loss)
            diffusion_losses.append(avg_diff_loss)
            if enable_physics:
                physics_losses.append(avg_phy_loss)
                current_physics_weight = get_physics_weight(epoch, epochs, physics_weight)
                physics_weights_log.append(current_physics_weight)

            print(f"\n📊 Epoch {epoch + 1}/{epochs}:")
            print(f"  总损失: {avg_loss:.6f}")
            print(f"  扩散损失: {avg_diff_loss:.6f}")
            if enable_physics:
                print(f"  物理损失: {avg_phy_loss:.6f}")
                print(f"  物理权重: {current_physics_weight:.4f}")
                print(f"  连续性损失: {avg_cont_loss:.6f}")
                print(f"  边界损失: {avg_bound_loss:.6f}")
            scheduler.step()
        if (epoch + 1) % save_every == 0:
            if enable_physics:
                save_path = os.path.join(SAVE_DIR, f"physics_cylinder_model_epoch_{epoch + 1}.pt")
            else:
                save_path = os.path.join(SAVE_DIR, f"cylinder_model_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': avg_loss,
                'diffusion_loss': avg_diff_loss,
                'physics_loss': avg_phy_loss if enable_physics else 0,
                'enable_physics': enable_physics,
                'physics_weight': physics_weight,
                'physics_weights_log': physics_weights_log if enable_physics else [],
            }, save_path)
            print(f"💾 检查点已保存: {save_path}")

    if enable_physics:
        final_save_path = os.path.join(SAVE_DIR, "physics_cylinder_model_final.pt")
    else:
        final_save_path = os.path.join(SAVE_DIR, "cylinder_model_final.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"\n💾 最终模型已保存: {final_save_path}")

    # 绘图部分（略，与原代码相同）
    plt.figure(figsize=(15, 5))
    if enable_physics and physics_losses:
        plt.subplot(1, 3, 1)
    else:
        plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, label='总损失')
    plt.plot(range(1, len(diffusion_losses) + 1), diffusion_losses, 'r--', linewidth=1.5, label='扩散损失')
    if enable_physics and physics_losses:
        plt.plot(range(1, len(physics_losses) + 1), physics_losses, 'g-.', linewidth=1.5, label='物理损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)

    if enable_physics and physics_losses:
        plt.subplot(1, 3, 2)
        x = range(1, len(losses) + 1)
        plt.plot(x, losses, 'b-', linewidth=2, label='总损失')
        plt.plot(x, diffusion_losses, 'r--', linewidth=1.5, label='扩散损失')
        plt.plot(x, physics_losses, 'g-.', linewidth=1.5, label='物理损失')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('对数尺度损失曲线')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        if physics_weights_log:
            plt.plot(range(1, len(physics_weights_log) + 1), physics_weights_log, 'm-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('物理权重')
            plt.title('物理约束权重变化')
            plt.grid(True)

    plt.tight_layout()
    if enable_physics:
        fig_name = os.path.join(SAVE_DIR, 'physics_cylinder_training_loss_corrected.png')
    else:
        fig_name = os.path.join(SAVE_DIR, 'cylinder_training_loss.png')
    plt.savefig(fig_name, dpi=120, bbox_inches='tight')
    print(f"📈 损失曲线已保存: {fig_name}")

    loss_data_path = os.path.join(SAVE_DIR, 'cylinder_loss_data_corrected.npz')
    if enable_physics and physics_losses:
        np.savez(loss_data_path,
                 total_loss=np.array(losses),
                 diffusion_loss=np.array(diffusion_losses),
                 physics_loss=np.array(physics_losses),
                 physics_weights=np.array(physics_weights_log))
    else:
        np.savez(loss_data_path,
                 total_loss=np.array(losses),
                 diffusion_loss=np.array(diffusion_losses))
    print(f"📊 损失数据已保存: {loss_data_path}")

    print("\n🎉 训练完成!")
    return model, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练带物理约束的条件扩散模型（修正版）')
    parser.add_argument('--no_physics', action='store_true', help='不使用物理约束')
    parser.add_argument('--physics_weight', type=float, default=0.1, help='最终物理约束权重')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_every', type=int, default=5, help='每多少轮保存一次')
    parser.add_argument('--data_path', type=str, default=None, help='数据文件路径')
    parser.add_argument('--resume', type=str, default=None, help='预训练模型路径（.pt文件）')

    args = parser.parse_args()

    if args.data_path:
        DATA_PATH = args.data_path
        print(f"🎯 使用指定的数据路径: {DATA_PATH}")

    enable_physics = not args.no_physics
    train_with_physics(
        enable_physics=enable_physics,
        physics_weight=args.physics_weight,
        save_every=args.save_every,
        resume_path=args.resume,
        epochs = args.epochs
    )
