# create_dummy_data.py
import numpy as np
import os

# 创建数据目录结构
base_dir = "data/venturi/train/geometry_0"
os.makedirs(base_dir, exist_ok=True)

# 创建一个虚拟的4通道流场数据 [C, H, W] = [4, 64, 64]
dummy_flow = np.random.randn(4, 64, 64).astype(np.float32)
np.save(os.path.join(base_dir, "dummy_sample.npy"), dummy_flow)
print(f"虚拟数据已保存至: {base_dir}/dummy_sample.npy")