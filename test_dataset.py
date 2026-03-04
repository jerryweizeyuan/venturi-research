# test_dataset.py
import sys

sys.path.append('.')

# 测试VenturiDataset
try:
    from venturi_dataset import VenturiDataset

    print("✅ VenturiDataset 导入成功")

    # 创建数据集实例
    dataset = VenturiDataset("data/venturi", img_size=64, mode="train")
    print(f"✅ 数据集创建成功，大小: {len(dataset)}")

    # 获取一个样本
    sample = dataset[0]
    print(f"✅ 样本获取成功")
    print(f"   流场数据形状: {sample['flow'].shape}")
    print(f"   标签: {sample['label']}")
    print(f"   几何名称: {sample['geometry']}")

    # 检查数据类型
    print(f"   数据类型: {sample['flow'].dtype}")
    print(f"   数据范围: [{sample['flow'].min():.3f}, {sample['flow'].max():.3f}]")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback

    traceback.print_exc()