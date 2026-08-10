import torch

# PyTorch 张量运算

# 准备两个张量用于演示
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print("a:\n", a)
print("b:\n", b)

# 1. 逐元素算术运算（形状必须相同，或可广播）
print("\n1. 逐元素算术运算:")
print("a + b:\n", a + b)          # 加法
print("a - b:\n", a - b)          # 减法
print("a * b:\n", a * b)          # 逐元素乘法（不是矩阵乘法！）
print("a / b:\n", a / b)          # 除法
print("a ** 2:\n", a ** 2)        # 幂运算
print("a % 2:\n", a % 2)          # 取模

# 也可以使用函数形式：torch.add(a, b), torch.mul(a, b) 等

# 2. 矩阵乘法（线性代数中的矩阵乘法）
print("\n2. 矩阵乘法:")
print("a @ b:\n", a @ b)                    # 运算符形式
print("torch.matmul(a, b):\n", torch.matmul(a, b))  # 函数形式，两者等价

# 3. 标量运算（广播）
print("\n3. 标量运算:")
print("a + 10:\n", a + 10)
print("a * 2:\n", a * 2)

# 4. 广播机制（不同形状自动扩展）
print("\n4. 广播机制:")
row = torch.tensor([10.0, 20.0])  # 形状 [2]
print("a + row（逐行加）:\n", a + row)  # [2,2] + [2] -> [2,2]
col = torch.tensor([[100.0], [200.0]])  # 形状 [2,1]
print("a + col（逐列加）:\n", a + col)

# 5. 归约运算（求和、均值、最值）
print("\n5. 归约运算:")
print("a.sum():", a.sum())              # 所有元素求和
print("a.sum(dim=0):", a.sum(dim=0))    # 按列求和（压缩第 0 维）
print("a.sum(dim=1):", a.sum(dim=1))    # 按行求和
print("a.mean():", a.mean())            # 均值
print("a.max():", a.max())              # 最大值
print("a.min():", a.min())              # 最小值
print("a.max(dim=1):", a.max(dim=1))    # 每行最大值（返回 values, indices）

# 6. 比较运算（返回布尔张量）
print("\n6. 比较运算:")
print("a > 2:\n", a > 2)
print("a == b:\n", a == b)

# 7. 常用数学函数
print("\n7. 常用数学函数:")
x = torch.tensor([1.0, 2.0, 3.0])
print("exp:", torch.exp(x))      # 指数
print("log:", torch.log(x))      # 自然对数
print("sqrt:", torch.sqrt(x))    # 平方根
print("abs:", torch.abs(torch.tensor([-1.0, 2.0])))  # 绝对值
print("sigmoid:", torch.sigmoid(x))  # Sigmoid 激活

# 8. 累加/累乘（注意：会改变原张量，就地操作）
print("\n8. 就地操作 (in-place):")
y = torch.tensor([1.0, 2.0])
y.add_(10)  # 带下划线后缀的是就地操作，修改原张量
print("y.add_(10) 后 y:", y)

# 9. 类型转换与运算的混合（整数除法）
print("\n9. 整数除法:")
int_a = torch.tensor([7, 8, 9])
print("int_a // 2:", int_a // 2)  # 整除
print("int_a % 2:", int_a % 2)    # 取余