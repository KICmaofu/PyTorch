#完成设备迁移(CPU/GPU)
import torch

#查看GPU是否可用及数量
print(torch.cuda.is_available())  #True表示可以使用GPU
print(torch.cuda.device_count())  #GPU数量
print("---------------------------")

#查看张量所在设备
x = torch.tensor([1, 2, 3])
print(x.device)  #默认在cpu上
print("---------------------------")

#迁移到GPU(.to方法)
x_gpu = x.to("cuda")
print(x_gpu.device)
print("---------------------------")

#迁移到GPU(.cuda方法)
x_gpu2 = x.cuda()
print(x_gpu2.device)
print("---------------------------")

#迁移回CPU(.cpu方法)
x_cpu = x_gpu.cpu()
print(x_cpu.device)
print("---------------------------")

#创建时直接指定设备
y = torch.tensor([1, 2, 3], device="cuda")
print(y.device)
z = torch.zeros(2, 3, device="cpu")
print(z.device)
print("---------------------------")

#指定GPU编号(多卡时使用)
c = torch.tensor([1, 2, 3], device="cuda:0")
print(c.device)
print("---------------------------")

#推荐写法:先定义device,代码在有无GPU的机器上都能跑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.tensor([1, 2, 3]).to(device)
print(d.device)
print("---------------------------")

#不同设备上的张量不能直接运算
a = torch.tensor([1, 2, 3], device="cpu")
b = torch.tensor([1, 2, 3], device="cuda")
# print(a + b)  #报错:Expected all tensors to be on the same device
print("---------------------------")

#GPU张量转Python标量(item会自动回到CPU)
s = torch.tensor(3.14, device="cuda")
print(s.item())
print("---------------------------")

#GPU张量转numpy必须先.cpu()
n = x_gpu.cpu().numpy()
print(n, type(n))
print("---------------------------")

#模型迁移到GPU
model = torch.nn.Linear(3, 1)
model = model.to("cuda")
print(next(model.parameters()).device)
print("---------------------------")

#查看显存占用(单位MB)
print(torch.cuda.memory_allocated() / 1024 ** 2)
