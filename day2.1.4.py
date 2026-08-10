import torch

# PyTorch 张量索引操作

# 准备一个二维张量用于演示（3 行 4 列）
x = torch.arange(12).reshape(3, 4)
print("x:\n", x)

# 1. 单个元素索引（下标从 0 开始）
print("\n1. 单个元素索引:")
print("x[0, 0]:", x[0, 0])      # 第 0 行第 0 列
print("x[1, 2]:", x[1, 2])      # 第 1 行第 2 列
print("x[2, 3]:", x[2, 3])      # 第 2 行第 3 列

# 2. 负索引（从末尾开始数，-1 表示最后一个）
print("\n2. 负索引:")
print("x[-1]:", x[-1])            # 最后一行
print("x[-1, -1]:", x[-1, -1])    # 最后一个元素
print("x[-2, -3]:", x[-2, -3])    # 倒数第 2 行第 1 列

# 3. 切片索引（左闭右开，start:end）
print("\n3. 切片索引:")
print("x[0]:", x[0])                # 第 0 行（返回一维张量）
print("x[0, 1:3]:", x[0, 1:3])      # 第 0 行第 1~2 列
print("x[:, 2]:", x[:, 2])          # 所有行的第 2 列
print("x[:2, :2]:\n", x[:2, :2])    # 前两行前两列
print("x[1:, :]:\n", x[1:, :])      # 从第 1 行到最后

# 4. 步长切片（start:end:step）
print("\n4. 步长切片:")
print("x[::2]:\n", x[::2])          # 每隔一行取一行
print("x[:, ::2]:\n", x[:, ::2])    # 每隔一列取一列
print("torch.flip(x, dims=[0]):\n", torch.flip(x, dims=[0]))        # 行倒序（PyTorch 不支持 x[::-1]）
print("torch.flip(x, dims=[0, 1]):\n", torch.flip(x, dims=[0, 1]))  # 全部倒序

# 5. 花式索引（用整数列表或张量指定位置）
print("\n5. 花式索引:")
print("x[[0, 2]]:\n", x[[0, 2]])          # 取第 0 行和第 2 行
print("x[[0, 2], [1, 3]]:", x[[0, 2], [1, 3]])  # 分别取 (0,1) 和 (2,3) 两个元素
idx = torch.tensor([0, 1])
print("x[idx]:\n", x[idx])                # 用张量做索引
print("torch.index_select(x, 0, idx):\n", torch.index_select(x, 0, idx))  # 函数形式

# 6. 布尔掩码索引（返回满足条件的元素，结果是一维）
print("\n6. 布尔掩码索引:")
mask = x > 5
print("mask:\n", mask)
print("x[mask]:", x[mask])
print("x[x % 2 == 0]:", x[x % 2 == 0])    # 取出所有偶数
print("x[x < 3]:", x[x < 3])              # 取出所有小于 3 的元素

# 7. torch.where 条件选择
print("\n7. torch.where:")
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])
print("torch.where(a > 1, a, b):", torch.where(a > 1, a, b))  # 条件为真取 a，否则取 b

# 8. 视图 vs 副本（重要！）
print("\n8. 视图 vs 副本:")
x1 = torch.arange(12).reshape(3, 4)
slice_view = x1[0:2]          # 切片是视图：与 x1 共享底层数据
slice_view[0, 0] = 999
print("修改切片后 x1:\n", x1)  # 原张量跟着变

x2 = torch.arange(12).reshape(3, 4)
fancy_copy = x2[[0, 1]]       # 花式索引是副本：数据独立
fancy_copy[0, 0] = -1
print("修改花式索引结果后 x2:\n", x2)  # 原张量不变

# 9. 索引赋值（修改指定位置的值）
print("\n9. 索引赋值:")
y = torch.zeros(3, 4)
y[:, 1] = 7         # 所有行的第 1 列设为 7
y[0, 0] = 100       # 单个元素赋值
y[y == 0] = 1       # 所有为 0 的位置设为 1
print("y:\n", y)

# 10. 三维张量的索引与省略号
print("\n10. 三维张量索引:")
t3 = torch.arange(24).reshape(2, 3, 4)  # 2 个 3x4 矩阵
print("t3:\n", t3)
print("t3[0]:\n", t3[0])          # 第一个矩阵
print("t3[0, 1]:", t3[0, 1])      # 第一个矩阵的第 1 行
print("t3[0, 1, 2]:", t3[0, 1, 2])  # 单个元素
print("t3[..., 2]:", t3[..., 2])  # ... 省略中间维度，等价于 t3[:, :, 2]
print("t3[:, 1:, :]:\n", t3[:, 1:, :])  # 每个矩阵从第 1 行开始

# 11. 其他常用索引相关操作
print("\n11. 常用索引相关操作:")
print("x.diagonal():", x.diagonal())          # 主对角线元素
print("x.nonzero():\n", x.nonzero())          # 非零元素的位置坐标
print("x.max(dim=1):", x.max(dim=1))          # 每行最大值（返回 values, indices）