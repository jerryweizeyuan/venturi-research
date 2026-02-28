# test_dataset.py
from cfd_dataset import CFDDataset

# 测试新数据集
dataset = CFDDataset(
    flow_dir="E:/CFD_Data/toy_cylinder/flow_fields",
    geom_path="E:/CFD_Data/toy_cylinder/geometry_masks/cylinder.pt",
    num_snapshots=8
)

flow, label = dataset[0]
print(f"✅ 数据加载成功")
print(f"   流场形状: {flow.shape}")  # 应该是 [4, 64, 64]
print(f"   标签: {label}")          # 应该是 tensor(0)

# 验证形状
if flow.shape == (4, 64, 64):
    print("🎉 形状正确！")
else:
    print(f"⚠️ 期望形状 [4, 64, 64]，实际得到 {list(flow.shape)}")