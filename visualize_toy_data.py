import torch
import matplotlib.pyplot as plt

# 加载数据
geom = torch.load("E:/CFD_Data/toy_cylinder/geometry_masks/cylinder.pt")
flow = torch.load("E:/CFD_Data/toy_cylinder/flow_fields/snapshot_0000.pt")

print(f"几何掩码形状: {geom.shape}")
print(f"流场形状: {flow.shape}")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 几何掩码
axes[0, 0].imshow(geom[0], cmap='gray')
axes[0, 0].set_title('Geometry Mask (0=fluid, 1=solid)')

# 流场的4个通道
channels = ['Velocity u', 'Velocity v', 'Pressure p', 'Vorticity ω']
for i in range(4):
    row, col = (i+1)//3, (i+1)%3
    im = axes[row, col].imshow(flow[i], cmap='RdBu_r')
    axes[row, col].set_title(f'{channels[i]}')
    plt.colorbar(im, ax=axes[row, col])

plt.tight_layout()
plt.savefig('toy_data_visualization.png')
plt.show()
print("可视化已保存为 toy_data_visualization.png")