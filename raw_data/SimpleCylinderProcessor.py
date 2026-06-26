"""
简洁版圆柱CFD数据处理
专门用于物理约束扩散模型训练
根据NaN值自动生成掩码，输出单个训练文件
"""

import numpy as np
import os
import re
import matplotlib.pyplot as plt
import time


class SimpleCylinderProcessor:
    """简洁版圆柱处理器 - 基于您的原始COMSOL格式"""

    def __init__(self, output_dir="simple_cylinder_data"):
        """初始化"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"简洁版圆柱CFD处理器 - 输出目录: {output_dir}")

    def _calculate_vorticity_simple(self, u_array, v_array, dx, dy):
        """计算涡量 - 简单版本"""
        # 先处理NaN
        u_clean = np.nan_to_num(u_array, nan=0.0)
        v_clean = np.nan_to_num(v_array, nan=0.0)

        # 计算梯度
        du_dy, du_dx = np.gradient(u_clean, dy, dx)
        dv_dy, dv_dx = np.gradient(v_clean, dy, dx)

        # 涡量 = dv/dx - du/dy
        omega = dv_dx - du_dy

        return omega

    def process_single_file(self, file_path):
        """处理单个COMSOL文件 - 完全基于您的格式"""
        print(f"处理文件: {os.path.basename(file_path)}")

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 查找数据开始位置
            grid_start = None
            u_start = None
            v_start = None
            p_start = None

            for i, line in enumerate(lines):
                if '% Grid' in line:
                    grid_start = i + 1
                elif '% u' in line and u_start is None:
                    u_start = i + 1
                elif '% v' in line and v_start is None:
                    v_start = i + 1
                elif '% p' in line and p_start is None:
                    p_start = i + 1

            if grid_start is None:
                raise ValueError("找不到Grid标记")

            # 读取x坐标
            x_coords = np.array([float(x) for x in lines[grid_start].strip().split()])

            # 读取y坐标
            y_coords = np.array([float(y) for y in lines[grid_start + 1].strip().split()])

            nx = len(x_coords)
            ny = len(y_coords)

            # 读取u数据
            u_data = []
            for i in range(u_start, u_start + ny):
                row = [float(x) for x in lines[i].strip().split()]
                u_data.extend(row)
            u_array = np.array(u_data).reshape(ny, nx)

            # 读取v数据
            v_data = []
            for i in range(v_start, v_start + ny):
                row = [float(x) for x in lines[i].strip().split()]
                v_data.extend(row)
            v_array = np.array(v_data).reshape(ny, nx)

            # 读取p数据
            p_data = []
            for i in range(p_start, p_start + ny):
                row = [float(x) for x in lines[i].strip().split()]
                p_data.extend(row)
            p_array = np.array(p_data).reshape(ny, nx)

            print(f"  网格: {nx}×{ny}")
            print(f"  u形状: {u_array.shape}, NaN: {np.isnan(u_array).sum()}")
            print(f"  v形状: {v_array.shape}, NaN: {np.isnan(v_array).sum()}")
            print(f"  p形状: {p_array.shape}, NaN: {np.isnan(p_array).sum()}")

            # 步骤1: 根据NaN值生成掩码
            # 流体区域 = 非NaN区域 = 1
            # 固体区域 = NaN区域 = 0
            mask = (~np.isnan(u_array)).astype(np.float32)

            # 统计掩码信息
            fluid_ratio = mask.mean()
            solid_ratio = 1 - fluid_ratio
            print(f"  掩码: 流体={fluid_ratio:.1%}, 固体={solid_ratio:.1%}")

            # 步骤2: 计算涡量
            if len(x_coords) > 1:
                dx = x_coords[1] - x_coords[0]
            else:
                dx = 1.0

            if len(y_coords) > 1:
                dy = y_coords[1] - y_coords[0]
            else:
                dy = 1.0

            if abs(dx) < 1e-10:
                dx = 1e-10
            if abs(dy) < 1e-10:
                dy = 1e-10

            omega_array = self._calculate_vorticity_simple(u_array, v_array, dx, dy)
            nan_mask = np.isnan(u_array)  # 找到NaN位置（圆柱区域）
            omega_array[nan_mask] = 0.0  # 将这些位置的涡量设为0
            print(f"  涡量: 将{nan_mask.sum()}个圆柱点的涡量设为0")

            # 步骤3: 清理NaN值（固体区域设为0）
            u_clean = np.nan_to_num(u_array, nan=0.0)
            v_clean = np.nan_to_num(v_array, nan=0.0)
            p_clean = np.nan_to_num(p_array, nan=0.0)
            omega_clean = np.nan_to_num(omega_array, nan=0.0)

            # 确保固体区域为0
            solid_mask = mask < 0.5
            u_clean[solid_mask] = 0.0
            v_clean[solid_mask] = 0.0
            p_clean[solid_mask] = 0.0
            omega_clean[solid_mask] = 0.0

            # 步骤4: 组合数据
            # 流场: (4, H, W) = [u, v, p, ω]
            flow_fields = np.stack([u_clean, v_clean, p_clean, omega_clean], axis=0)

            # 掩码: (1, H, W)
            geometry_masks = mask[np.newaxis, :, :]

            return flow_fields, geometry_masks

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return None, None

    def process_all_files(self, file_patterns=None):
        """处理所有圆柱数据文件，生成单个训练文件"""
        if file_patterns is None:
            # 自动查找所有包含"cylinder"的txt文件
            all_files = [f for f in os.listdir('.') if f.endswith('.txt')]
            cylinder_files = []

            for f in all_files:
                f_lower = f.lower()
                if any(pattern in f_lower for pattern in ['cylinder', 'flow_uin']):
                    cylinder_files.append(f)
        else:
            cylinder_files = []
            for pattern in file_patterns:
                if os.path.exists(pattern):
                    cylinder_files.append(pattern)

        if not cylinder_files:
            print("⚠️ 警告: 没有找到圆柱数据文件!")
            print("请确保当前目录有圆柱数据文件，文件名包含: cylinder, circ, round, case")
            return None

        print(f"\n找到 {len(cylinder_files)} 个圆柱数据文件:")
        for f in cylinder_files:
            print(f"  • {f}")

        # 处理所有文件
        all_flow = []
        all_masks = []

        for i, file_path in enumerate(cylinder_files):
            print(f"\n[{i + 1}/{len(cylinder_files)}] 处理: {file_path}")

            flow, mask = self.process_single_file(file_path)

            if flow is not None and mask is not None:
                all_flow.append(flow)
                all_masks.append(mask)
                print(f"  ✅ 处理成功")
            else:
                print(f"  ❌ 处理失败，跳过")

        if not all_flow:
            print("\n❌ 错误: 没有成功处理任何文件!")
            return None

        # 检查所有数据形状是否一致
        shapes = [f.shape for f in all_flow]
        unique_shapes = set(shapes)

        if len(unique_shapes) > 1:
            print(f"⚠️ 警告: 数据形状不一致: {shapes}")
            print("将裁剪到最小公共形状...")

            # 找到最小公共形状
            min_c = min([s[0] for s in shapes])  # 通道数
            min_h = min([s[1] for s in shapes])  # 高度
            min_w = min([s[2] for s in shapes])  # 宽度
            target_shape = (min_c, min_h, min_w)

            print(f"目标形状: {target_shape}")

            # 裁剪所有数据
            all_flow_cropped = []
            all_masks_cropped = []

            for flow, mask in zip(all_flow, all_masks):
                flow_cropped = flow[:, :min_h, :min_w]
                mask_cropped = mask[:, :min_h, :min_w]
                all_flow_cropped.append(flow_cropped)
                all_masks_cropped.append(mask_cropped)

            all_flow = all_flow_cropped
            all_masks = all_masks_cropped

        # 堆叠数据
        flow_fields = np.stack(all_flow, axis=0)  # (N, 4, H, W)
        geometry_masks = np.stack(all_masks, axis=0)  # (N, 1, H, W)

        print(f"\n✅ 数据处理完成!")
        print(f"  总样本数: {len(all_flow)}")
        print(f"  流场形状: {flow_fields.shape} (样本数, 通道数, 高度, 宽度)")
        print(f"  掩码形状: {geometry_masks.shape} (样本数, 1, 高度, 宽度)")

        # 最终数据质量检查
        print(f"\n数据质量检查:")
        print(f"  NaN总数: {np.isnan(flow_fields).sum()}")
        print(f"  Inf总数: {np.isinf(flow_fields).sum()}")
        print(f"  掩码均值: {geometry_masks.mean():.4f} (流体占比)")

        # 各通道统计
        channel_names = ['u速度', 'v速度', '压力', '涡量']
        for i, name in enumerate(channel_names):
            channel_data = flow_fields[:, i]
            print(f"  {name}: [{channel_data.min():.6f}, {channel_data.max():.6f}]")

        # 保存为单个训练文件
        output_path = os.path.join(self.output_dir, "cylinder_training_data.npz")
        np.savez_compressed(
            output_path,
            flow_fields=flow_fields.astype(np.float32),
            geometry_masks=geometry_masks.astype(np.float32)
        )

        print(f"\n💾 训练数据保存: {output_path}")

        # 生成可视化
        self._visualize_sample(flow_fields[0], geometry_masks[0], output_path.replace('.npz', '_sample0.png'))

        # 生成统计信息
        self._create_data_stats(flow_fields, geometry_masks, output_path.replace('.npz', '_stats.txt'))

        return output_path

    def _visualize_sample(self, flow_sample, mask_sample, save_path):
        """可视化一个样本"""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        u, v, p, omega = flow_sample
        mask = mask_sample[0]  # 去掉通道维度

        # 1. 掩码
        ax = axes[0, 0]
        im0 = ax.imshow(mask, cmap='binary', vmin=0, vmax=1)
        ax.set_title('几何掩码\n(白色=流体, 黑色=固体)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

        # 2. u速度
        ax = axes[0, 1]
        im1 = ax.imshow(u, cmap='coolwarm')
        ax.set_title('u速度 (x方向)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        # 3. v速度
        ax = axes[0, 2]
        im2 = ax.imshow(v, cmap='coolwarm')
        ax.set_title('v速度 (y方向)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

        # 4. 压力
        ax = axes[1, 0]
        im3 = ax.imshow(p, cmap='RdBu_r')
        ax.set_title('压力 p')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

        # 5. 涡量
        ax = axes[1, 1]
        im4 = ax.imshow(omega, cmap='RdBu_r')
        ax.set_title('涡量 ω')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

        # 6. 速度矢量
        ax = axes[1, 2]
        speed = np.sqrt(u ** 2 + v ** 2)  # 修正：应该是u**2 + v**2，不是u2 + v2
        h, w = u.shape
        stride_x = max(1, h // 20)
        stride_y = max(1, w // 20)

        y, x = np.meshgrid(np.arange(0, w, stride_x), np.arange(0, h, stride_y))
        u_subsample = u[::stride_y, ::stride_x]
        v_subsample = v[::stride_y, ::stride_x]

        im5 = ax.imshow(speed, cmap='viridis', alpha=0.7)
        ax.quiver(x, y, u_subsample, v_subsample, color='white',
                  scale=20, width=0.002, headwidth=3, headlength=4)
        ax.set_title('速度矢量场')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im5, ax=ax, fraction=0.046, pad=0.04, label='速度幅值')

        plt.suptitle(f'训练数据示例 (流体占比: {mask.mean():.1%})', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 示例可视化: {save_path}")

    def _create_data_stats(self, flow_fields, geometry_masks, save_path):
        """创建数据统计信息"""
        with open(save_path, 'w') as f:
            f.write("圆柱CFD训练数据统计信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("数据集信息:\n")
            f.write(f"  样本数量: {flow_fields.shape[0]}\n")
            f.write(f"  流场形状: {flow_fields.shape} (N, C, H, W)\n")
            f.write(f"  掩码形状: {geometry_masks.shape} (N, 1, H, W)\n")
            f.write(f"  通道顺序: 0=u, 1=v, 2=p, 3=ω\n\n")

            f.write("掩码统计:\n")
            f.write(f"  流体占比: {geometry_masks.mean():.4f}\n")
            f.write(f"  固体占比: {1 - geometry_masks.mean():.4f}\n")
            f.write(f"  掩码值范围: [{geometry_masks.min()}, {geometry_masks.max()}]\n\n")

            f.write("各通道统计:\n")
            channel_names = ['u速度', 'v速度', '压力', '涡量']
            for i, name in enumerate(channel_names):
                channel_data = flow_fields[:, i]
                f.write(f"  {name}:\n")
                f.write(f"    最小值: {channel_data.min():.6f}\n")
                f.write(f"    最大值: {channel_data.max():.6f}\n")
                f.write(f"    平均值: {channel_data.mean():.6f}\n")
                f.write(f"    标准差: {channel_data.std():.6f}\n")
                f.write(f"    NaN数量: {np.isnan(channel_data).sum()}\n")

            f.write(f"\n数据质量:\n")
            f.write(f"  总NaN数量: {np.isnan(flow_fields).sum()}\n")
            f.write(f"  总Inf数量: {np.isinf(flow_fields).sum()}\n")
            f.write(f"  数据类型: {flow_fields.dtype}\n")

        print(f"📈 数据统计: {save_path}")


def load_training_data(npz_path, normalize=True, split_ratio=0.8):
    """加载训练数据并进行预处理"""
    print(f"\n加载训练数据: {npz_path}")

    data = np.load(npz_path)
    flow_fields = data['flow_fields']  # (N, 4, H, W)
    geometry_masks = data['geometry_masks']  # (N, 1, H, W)
    data.close()

    print(f"  原始形状: {flow_fields.shape}")
    print(f"  掩码形状: {geometry_masks.shape}")

    # 转换为PyTorch张量
    import torch
    flow_tensor = torch.FloatTensor(flow_fields)
    mask_tensor = torch.FloatTensor(geometry_masks)

    # 标准化
    if normalize:
        # 计算每个通道的均值和标准差
        flow_means = []
        flow_stds = []

        for i in range(flow_tensor.shape[1]):  # 对每个通道
            channel_data = flow_tensor[:, i]
            mean = channel_data.mean()
            std = channel_data.std()
            flow_means.append(mean.item())
            flow_stds.append(std.item())

            if std < 1e-8:
                std = 1.0

            flow_tensor[:, i] = (channel_data - mean) / std

        print(f"  标准化参数:")
        print(f"    通道均值: {flow_means}")
        print(f"    通道标准差: {flow_stds}")

    # 分割数据集
    n_samples = len(flow_tensor)
    train_size = int(n_samples * split_ratio)
    indices = torch.randperm(n_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_flow = flow_tensor[train_indices]
    train_mask = mask_tensor[train_indices]
    val_flow = flow_tensor[val_indices]
    val_mask = mask_tensor[val_indices]

    print(f"  训练集: {len(train_flow)} 样本")
    print(f"  验证集: {len(val_flow)} 样本")

    return (train_flow, train_mask), (val_flow, val_mask)


def create_simple_model_input(mask, device='cpu'):
    """为扩散模型创建输入"""
    import torch

    # 基本检查
    if mask.dim() == 3:  # (1, H, W)
        mask = mask.unsqueeze(0)  # -> (1, 1, H, W)
    elif mask.dim() == 2:  # (H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

    # 移动到设备
    mask = mask.to(device)

    # 创建带噪声的初始流场
    batch_size, _, h, w = mask.shape

    # 初始化流场为0
    flow = torch.zeros(batch_size, 4, h, w, device=device)

    # 添加随机噪声
    noise = torch.randn_like(flow) * 0.1
    flow = flow + noise

    # 确保固体区域为0
    solid_mask = mask < 0.5
    flow[:, 0, solid_mask] = 0.0  # u
    flow[:, 1, solid_mask] = 0.0  # v
    flow[:, 2, solid_mask] = 0.0  # p
    flow[:, 3, solid_mask] = 0.0  # ω

    return flow, mask


def main():
    """主函数"""
    print("=" * 60)
    print("简洁版圆柱CFD数据处理")
    print("专门用于物理约束扩散模型训练")
    print("=" * 60)

    # 创建处理器
    processor = SimpleCylinderProcessor(output_dir="cylinder_training_data")

    # 处理所有文件
    # 可以指定文件列表，或者自动查找
    file_list = None  # 设为None自动查找

    # 如果您有特定文件，可以这样指定：
    # file_list = ["cylinder_Re100.txt", "cylinder_Re200.txt", "case1.txt"]

    npz_path = processor.process_all_files(file_list)

    if npz_path is None:
        print("\n❌ 处理失败!")
        return

    # 加载并检查数据
    print(f"\n{'=' * 60}")
    print("数据加载验证")
    print('=' * 60)

    data = np.load(npz_path)
    flow_fields = data['flow_fields']
    geometry_masks = data['geometry_masks']
    data.close()

    print(f"✅ 数据加载成功:")
    print(f"  文件: {npz_path}")
    print(f"  流场形状: {flow_fields.shape}")
    print(f"  掩码形状: {geometry_masks.shape}")
    print(f"  数据类型: {flow_fields.dtype}")
    print(f"  数据范围: [{flow_fields.min():.6f}, {flow_fields.max():.6f}]")
    print(f"  NaN检查: {np.isnan(flow_fields).any()}")
    print(f"  Inf检查: {np.isinf(flow_fields).any()}")

    # 检查通道顺序
    print(f"\n通道验证:")
    print(f"  通道0: u速度, 范围: [{flow_fields[:, 0].min():.6f}, {flow_fields[:, 0].max():.6f}]")
    print(f"  通道1: v速度, 范围: [{flow_fields[:, 1].min():.6f}, {flow_fields[:, 1].max():.6f}]")
    print(f"  通道2: 压力, 范围: [{flow_fields[:, 2].min():.6f}, {flow_fields[:, 2].max():.6f}]")
    print(f"  通道3: 涡量, 范围: [{flow_fields[:, 3].min():.6f}, {flow_fields[:, 3].max():.6f}]")

    # 检查掩码
    print(f"\n掩码验证:")
    unique_values = np.unique(geometry_masks)
    print(f"  掩码唯一值: {unique_values}")
    print(f"  流体占比: {geometry_masks.mean():.4f}")
    print(f"  固体占比: {1 - geometry_masks.mean():.4f}")

    # 创建使用说明
    readme_path = os.path.join(processor.output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# 圆柱CFD训练数据\n\n")
        f.write("## 文件格式\n")
        f.write("- `flow_fields`: 形状 (N, 4, H, W)\n")
        f.write("- `geometry_masks`: 形状 (N, 1, H, W)\n\n")
        f.write("## 通道顺序\n")
        f.write("0. u速度 (x方向)\n")
        f.write("1. v速度 (y方向)\n")
        f.write("2. 压力 p\n")
        f.write("3. 涡量 ω\n\n")
        f.write("## 掩码含义\n")
        f.write("- 1.0: 流体区域\n")
        f.write("- 0.0: 固体区域 (圆柱)\n\n")
        f.write("## 使用示例\n")
        f.write("```python\n")
        f.write("import numpy as np\n")
        f.write("import torch\n\n")
        f.write("# 加载数据\n")
        f.write("data = np.load('cylinder_training_data.npz')\n")
        f.write("flow_fields = data['flow_fields']  # (N, 4, H, W)\n")
        f.write("geometry_masks = data['geometry_masks']  # (N, 1, H, W)\n")
        f.write("data.close()\n\n")
        f.write("# 转换为PyTorch张量\n")
        f.write("flow_tensor = torch.FloatTensor(flow_fields)\n")
        f.write("mask_tensor = torch.FloatTensor(geometry_masks)\n\n")
        f.write("# 扩散模型输入\n")
        f.write("# mask: 几何掩码，1=流体，0=固体\n")
        f.write("# flow: 流场，需要在扩散过程中满足边界条件\n")
        f.write("```\n\n")
        f.write("## 数据统计\n")
        f.write(f"- 样本数: {flow_fields.shape[0]}\n")
        f.write(f"- 空间维度: {flow_fields.shape[2:]} (H, W)\n")
        f.write(f"- 流体占比: {geometry_masks.mean():.4f}\n")
        f.write(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n📄 使用说明: {readme_path}")
    print(f"\n{'=' * 60}")
    print("✅ 处理完成!")
    print(f"训练数据: {npz_path}")
    print(f"可以用于您的物理约束扩散模型训练")
    print("=" * 60)

    return npz_path


if __name__ == "__main__":
    # 运行主程序
    data_path = main()

    if data_path:
        print(f"\n下一步:")
        print(f"1. 在您的模型中使用: data = np.load('{data_path}')")
        print(f"2. flow_fields = data['flow_fields']  # (N, 4, H, W)")
        print(f"3. geometry_masks = data['geometry_masks']  # (N, 1, H, W)")
        print(f"4. 训练扩散模型: 输入=掩码, 输出=流场")
