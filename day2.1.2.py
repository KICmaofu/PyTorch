import torch

# PyTorch 张量元素类型转换

# 1. 查看张量类型
x = torch.tensor([1, 2, 3])  # 默认创建为 int64
print("1. 查看类型:")
print("x:", x, "dtype:", x.dtype)

y = torch.tensor([1.5, 2.5, 3.5])  # 默认创建为 float32
print("y:", y, "dtype:", y.dtype)

# 2. 创建时直接指定类型
print("\n2. 创建时指定 dtype:")
z = torch.tensor([1, 2, 3], dtype=torch.float64)
print("z:", z, "dtype:", z.dtype)

# 3. 使用 .to() 方法转换类型（最通用）
print("\n3. 使用 .to() 转换:")
a = torch.tensor([1.7, 2.2, 3.9])
print("原始:", a, a.dtype)
b = a.to(torch.int32)  # float -> int 会截断小数部分
print("转 int32:", b, b.dtype)
c = a.to(torch.float64)  # float32 -> float64
print("转 float64:", c, c.dtype)

# 4. 使用类型简写方法（最常用）
print("\n4. 类型简写方法:")
int_tensor = torch.tensor([1, 2, 3])
print("int -> float():", int_tensor.float(), int_tensor.float().dtype)
print("int -> long():", int_tensor.long(), int_tensor.long().dtype)
print("float -> int():", torch.tensor([1.9, 2.1]).int())  # 注意：截断而不是四舍五入
print("float -> bool():", torch.tensor([0.0, 1.0, -2.5]).bool())  # 非 0 即为 True
print("int -> double():", int_tensor.double(), int_tensor.double().dtype)

# 5. 转换时截断 vs 四舍五入的区别
print("\n5. 截断 vs 四舍五入:")
f = torch.tensor([2.6, 2.4, -2.6])
print("原始:", f)
print("int() 截断:", f.int())          # 直接去掉小数部分
print("round() 四舍五入:", f.round().int())  # 先四舍五入再转 int

# 6. 类型转换不影响原张量（返回新张量）
print("\n6. 转换返回新张量，原张量不变:")
original = torch.tensor([1.5, 2.5])
converted = original.int()
print("原张量:", original, original.dtype)
print("转换后:", converted, converted.dtype)

# 7. 完整类型对应表
print("\n7. 常见 dtype 对照:")
print("torch.float32 (float)  - 32位浮点数，默认浮点类型")
print("torch.float64 (double) - 64位浮点数，高精度")
print("torch.int64   (long)   - 64位整数，默认整数类型")
print("torch.int32   (int)    - 32位整数")
print("torch.int16   (short)  - 16位整数")
print("torch.uint8            - 无符号8位整数")
print("torch.bool             - 布尔类型 (True/False)")