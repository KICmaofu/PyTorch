
import torch

# 创建线性张量和随机张量
# 1. 线性张量（等差数列，torch.arange）
linear_tensor = torch.arange(12)  # 生成 0 到 11 的一维张量
print("线性张量 (arange):\n", linear_tensor)
print("Shape:", linear_tensor.shape)  # 形状为 torch.Size([12])
# 2. 随机张量（标准正态分布，torch.randn）
randn_tensor = torch.randn(3, 4)  # 3x4，均值为 0、标准差为 1
print("随机张量 (randn):\n", randn_tensor)
print("Shape:", randn_tensor.shape)
# 3. 随机张量（[0, 1) 均匀分布，torch.rand）
rand_tensor = torch.rand(3, 4)  # 3x4，元素值在 [0, 1) 之间
print("均匀随机张量 (rand):\n", rand_tensor)
# 4. 其他常用创建方式
zeros_tensor = torch.zeros((2, 3, 4))  # 全零张量
ones_tensor = torch.ones((2, 3, 4))    # 全一张量
print("全零张量:\n", zeros_tensor)
print("全一张量:\n", ones_tensor)
# 5. 线性张量改变形状
reshaped = linear_tensor.reshape(3, 4)  # 将一维线性张量变为 3x4
print("reshape 后的线性张量:\n", reshaped)