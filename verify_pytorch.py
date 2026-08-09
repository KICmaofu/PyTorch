"""PyTorch 安装验证脚本。

验证内容：
1. torch 导入与版本
2. CUDA / 设备可用性
3. 张量基本运算
4. 自动求导 (autograd)
5. 一个极简神经网络的前向/反向传播
"""

import sys
import torch
import torch.nn as nn


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    failures = 0

    # 1. 版本与环境
    section("1. 版本与环境")
    print(f"Python : {sys.version.split()[0]}")
    print(f"torch  : {torch.__version__}")
    print(f"编译 CUDA 版本 : {torch.version.cuda}")
    print(f"可用设备数     : {torch.cuda.device_count()}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA 可用       : {cuda_ok}")
    device = torch.device("cuda" if cuda_ok else "cpu")
    print(f"使用设备        : {device}")

    # 2. 张量基本运算
    section("2. 张量基本运算")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.ones(2, 2, device=device) * 2
    c = a @ b  # 矩阵乘法
    expected = torch.tensor([[6.0, 6.0], [14.0, 14.0]], device=device)
    print("a =\n", a)
    print("b =\n", b)
    print("a @ b =\n", c)
    if not torch.allclose(c, expected):
        print("[FAIL] 矩阵乘法结果不正确")
        failures += 1
    else:
        print("[OK] 矩阵乘法结果正确")

    # 3. 自动求导
    section("3. 自动求导 (autograd)")
    x = torch.tensor(3.0, requires_grad=True, device=device)
    y = x ** 2 + 2 * x + 1   # y = (x+1)^2
    y.backward()
    print(f"x = {x.item()}, y = x^2+2x+1 = {y.item()}")
    print(f"dy/dx (应为 2x+2 = {2*3+2}) = {x.grad.item()}")
    if abs(x.grad.item() - (2 * 3 + 2)) > 1e-5:
        print("[FAIL] autograd 计算错误")
        failures += 1
    else:
        print("[OK] autograd 计算正确")

    # 4. 极简神经网络
    section("4. 极简神经网络 (线性回归)")
    torch.manual_seed(0)
    # 目标: y = 2x + 0.5
    X = torch.linspace(-1, 1, 100, device=device).reshape(-1, 1)
    y_target = 2 * X + 0.5

    model = nn.Linear(1, 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y_target)
        loss.backward()
        opt.step()

    w = model.weight.item()
    b = model.bias.item()
    print(f"训练 300 轮后  loss = {loss.item():.6f}")
    print(f"拟合权重 w ≈ {w:.4f} (目标 2.0)")
    print(f"拟合偏置 b ≈ {b:.4f} (目标 0.5)")
    if loss.item() > 1e-3 or abs(w - 2.0) > 0.05 or abs(b - 0.5) > 0.05:
        print("[FAIL] 神经网络未收敛")
        failures += 1
    else:
        print("[OK] 神经网络收敛正确")

    # 5. GPU 验证（若可用）
    section("5. GPU 验证")
    if cuda_ok:
        g = torch.randn(1000, 1000, device="cuda")
        g = g @ g
        torch.cuda.synchronize()
        print(f"GPU 上 1000x1000 矩阵乘法完成，结果范数 = {g.norm().item():.4f}")
        print("[OK] GPU 计算正常")
    else:
        print("未检测到 CUDA，跳过 GPU 验证（CPU 安装同样可用）。")

    # 总结
    section("总结")
    if failures == 0:
        print("PyTorch 安装验证全部通过！")
        return 0
    print(f"验证失败 {failures} 项，请检查上方日志。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
