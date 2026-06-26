"""
修正版物理约束模块
为流体力学扩散模型提供物理约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstraintsCorrected:
    def __init__(self, loss_weights=None, dx=1.0 / 63.0, dy=1.0 / 63.0, device='cuda'):
        self.device = device
        self.dx = dx
        self.dy = dy
        if loss_weights is None:
            self.loss_weights = {
                'divergence': 0.1,
                'boundary': 1.0,
                'vorticity': 0.0,
                'energy': 0.0,
            }
        else:
            self.loss_weights = loss_weights

    def compute_continuity_loss(self, velocity_field, mask=None):
        u = velocity_field[:, 0:1]
        v = velocity_field[:, 1:2]
        dudx = (u[:, :, :, 1:] - u[:, :, :, :-1]) / self.dx
        dvdy = (v[:, :, 1:, :] - v[:, :, :-1, :]) / self.dy
        dudx = F.pad(dudx, (0, 1, 0, 0), mode='constant', value=0)
        dvdy = F.pad(dvdy, (0, 0, 0, 1), mode='constant', value=0)
        divergence = dudx + dvdy
        if mask is not None:
            divergence = divergence * mask
        loss = torch.mean(divergence ** 2)
        return loss

    def compute_boundary_loss(self, velocity_field, geometry_mask):
        solid_mask = 1.0 - geometry_mask
        velocity_magnitude = torch.sqrt(torch.sum(velocity_field ** 2, dim=1, keepdim=True))
        velocity_in_solid = velocity_magnitude * solid_mask
        boundary_loss = torch.mean(velocity_in_solid ** 2)
        return boundary_loss

    def get_adaptive_weights(self, epoch, total_epochs):
        progress = min(1.0, epoch / max(total_epochs, 1))
        adaptive_weights = {}
        for key, base_weight in self.loss_weights.items():
            if key == 'divergence':
                adaptive_weights[key] = base_weight * (0.5 + 0.5 * progress)
            elif key == 'boundary':
                adaptive_weights[key] = base_weight
            else:
                adaptive_weights[key] = base_weight
        return adaptive_weights

    def compute_total_loss(self, velocity_field, geometry_mask=None, epoch=0, total_epochs=100):
        loss_dict = {}
        if self.loss_weights.get('divergence', 0) > 0:
            loss_dict['divergence'] = self.compute_continuity_loss(velocity_field, geometry_mask)
        if self.loss_weights.get('boundary', 0) > 0 and geometry_mask is not None:
            loss_dict['boundary'] = self.compute_boundary_loss(velocity_field, geometry_mask)
        weights = self.get_adaptive_weights(epoch, total_epochs)
        total_loss = 0.0
        for key, loss in loss_dict.items():
            weight = weights.get(key, 0)
            if weight > 0:
                total_loss += weight * loss
        return total_loss, loss_dict, weights

    def __call__(self, velocity_field, geometry_mask=None, epoch=0, total_epochs=100):
        return self.compute_total_loss(velocity_field, geometry_mask, epoch, total_epochs)


class SimplePhysicsConstraints:
    """
    简化版物理约束，包含标准化处理，防止数值爆炸
    """
    def __init__(self, divergence_weight=0.1, boundary_weight=1.0, device='cpu'):
        self.device = device
        self.divergence_weight = divergence_weight
        self.boundary_weight = boundary_weight

    def compute_continuity_loss(self, velocity_field, dx=1.0, dy=1.0):
        u = velocity_field[:, 0:1]
        v = velocity_field[:, 1:2]
        # 标准化
        u_mean = u.mean(dim=[2, 3], keepdim=True)
        u_std = u.std(dim=[2, 3], keepdim=True) + 1e-8
        v_mean = v.mean(dim=[2, 3], keepdim=True)
        v_std = v.std(dim=[2, 3], keepdim=True) + 1e-8
        u_norm = (u - u_mean) / u_std
        v_norm = (v - v_mean) / v_std
        # 中心差分
        dudx = torch.zeros_like(u_norm)
        dudx[:, :, :, 1:-1] = (u_norm[:, :, :, 2:] - u_norm[:, :, :, :-2]) / (2 * dx)
        dvdy = torch.zeros_like(v_norm)
        dvdy[:, :, 1:-1, :] = (v_norm[:, :, 2:, :] - v_norm[:, :, :-2, :]) / (2 * dy)
        divergence = dudx + dvdy
        continuity_loss = torch.mean(divergence ** 2)
        return continuity_loss

    def compute_boundary_loss(self, velocity_field, geometry_mask):
        solid_mask = 1.0 - geometry_mask
        u = velocity_field[:, 0:1]
        v = velocity_field[:, 1:2]
        u_mean = u.mean(dim=[2, 3], keepdim=True)
        u_std = u.std(dim=[2, 3], keepdim=True) + 1e-8
        v_mean = v.mean(dim=[2, 3], keepdim=True)
        v_std = v.std(dim=[2, 3], keepdim=True) + 1e-8
        u_norm = (u - u_mean) / u_std
        v_norm = (v - v_mean) / v_std
        velocity_in_solid = torch.cat([u_norm, v_norm], dim=1) * solid_mask
        boundary_loss = torch.mean(velocity_in_solid ** 2)
        return boundary_loss

    def __call__(self, velocity_field, mask=None, epoch=0, total_epochs=100):
        continuity_loss = self.compute_continuity_loss(velocity_field)
        boundary_loss = self.compute_boundary_loss(velocity_field, mask) if mask is not None else torch.tensor(0.0, device=self.device)
        total_loss = self.divergence_weight * continuity_loss + self.boundary_weight * boundary_loss
        details = {
            'continuity': continuity_loss.item(),
            'boundary': boundary_loss.item(),
            'total_physics': total_loss.item()
        }
        return total_loss, details


# 测试函数
def test_corrected_physics():
    print("测试修正版物理约束模块...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    batch_size, channels, height, width = 2, 2, 64, 64
    velocity = torch.randn(batch_size, channels, height, width).to(device)
    geometry = torch.ones(batch_size, 1, height, width).to(device)
    geometry[:, :, 20:40, 20:40] = 0

    # 测试简化版本
    print("\n测试简化物理约束...")
    simple_physics = SimplePhysicsConstraints(divergence_weight=0.1, boundary_weight=1.0, device=device)
    total_loss, loss_dict = simple_physics(velocity, geometry, epoch=50, total_epochs=100)
    print(f"总损失: {total_loss.item():.6f}")
    print(f"连续性损失: {loss_dict['continuity']:.6f}")
    print(f"边界损失: {loss_dict['boundary']:.6f}")

    # 测试完整版本
    print("\n测试完整物理约束...")
    full_physics = PhysicsConstraintsCorrected(device=device)
    total_loss_full, loss_dict_full, weights_full = full_physics(velocity, geometry, epoch=50, total_epochs=100)
    print(f"总损失: {total_loss_full.item():.6f}")
    print(f"各项损失: {loss_dict_full}")
    print(f"使用的权重: {weights_full}")
    print("\n✅ 修正版物理约束模块测试完成!")


if __name__ == "__main__":
    test_corrected_physics()
