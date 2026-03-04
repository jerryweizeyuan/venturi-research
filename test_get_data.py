# test_get_data.py
import sys

sys.path.append('.')


# 模拟参数对象
class Args:
    dataset_path = "data/venturi"
    image_size = 64
    batch_size = 2
    num_classes = 1


args = Args()

# 测试get_data函数
try:
    from utils import get_data

    print("✅ get_data 函数导入成功")

    # 创建数据加载器
    dataloader = get_data(args)
    print(f"✅ 数据加载器创建成功")
    print(f"   类型: {type(dataloader)}")

    # 获取一个批次
    batch = next(iter(dataloader))
    print(f"✅ 批次获取成功")

    if isinstance(batch, dict):
        print(f"   批次大小: {batch['flow'].shape[0]}")
        print(f"   流场形状: {batch['flow'].shape}")
        print(f"   标签形状: {batch['label'].shape}")
    else:
        print(f"   批次结构: {type(batch)}")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback

    traceback.print_exc()