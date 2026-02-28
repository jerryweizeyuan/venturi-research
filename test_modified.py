# test_modified.py
import torch
from modules import UNet_conditional

print("测试修改后的模型...")

# 尝试创建接受4通道输入的模型
try:
    model = UNet_conditional(c_in=4, c_out=4, num_classes=1, device="cpu")
    print("✅ 模型创建成功！")
    print(f"  输入通道: 4 (CFD数据: u, v, p, ω)")
    print(f"  输出通道: 4")
    print(f"  类别数: 1 (暂时用数字标签)")

    # 测试一个CFD数据样本
    test_input = torch.randn(2, 4, 128, 128)  # 批次2, 4通道, 128x128
    t = torch.tensor([100, 200])  # 两个时间步
    y = torch.tensor([0, 0])  # 标签都为0（圆柱）

    output = model(test_input, t, y)
    print(f"✅ 前向传播成功！")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {output.shape}")

    if output.shape == test_input.shape:
        print("🎉 所有测试通过！模型已准备好处理CFD数据。")
    else:
        print(f"⚠️ 输出形状 {output.shape} 与输入形状 {test_input.shape} 不匹配")

except Exception as e:
    print(f"❌ 错误: {e}")
    print("请检查修改是否正确。")