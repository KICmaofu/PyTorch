#创建不同维度张量
import torch

#0维张量(标量):一个数
t0 = torch.tensor(5.)
print(t0)
print(t0.shape)  #torch.Size([])
print(t0.dim())  #0
print("---------------------------")

#1维张量(向量):一列数
t1 = torch.tensor([1, 2, 3])
print(t1)
print(t1.shape)  #torch.Size([3])
print(t1.dim())  #1
print("---------------------------")

#2维张量(矩阵):行x列
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t2)
print(t2.shape)  #torch.Size([2, 3])
print(t2.dim())  #2
print("---------------------------")

#3维张量:多张矩阵堆叠
t3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t3)
print(t3.shape)  #torch.Size([2, 2, 2])
print(t3.dim())  #3
print("---------------------------")

#4维张量:常见于图像数据(批量, 通道, 高, 宽)
t4 = torch.rand(2, 3, 4, 4)
print(t4.shape)  #torch.Size([2, 3, 4, 4])
print(t4.dim())  #4
print("---------------------------")

#用形状参数创建:参数个数就是维度数
print(torch.zeros(3).dim())        #1维
print(torch.ones(2, 3).dim())      #2维
print(torch.randn(2, 3, 4).dim())  #3维
print("---------------------------")

#维度属性汇总:dim()与ndim等价,shape与size()等价
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.dim(), x.ndim)
print(x.shape, x.size())
print(x.numel())  #元素总个数
print("---------------------------")

#改变形状:view/reshape不改变数据,只改变解释方式
y = torch.arange(12)
print(y)
print(y.view(3, 4))
print(y.reshape(2, 6))
print("---------------------------")

#view参数用-1自动推断
print(y.view(3, -1).shape)  #和view(3, 4)一样
print(y.view(-1, 6).shape)  #和reshape(2, 6)一样
print("---------------------------")

#增加维度:unsqueeze在指定位置插入大小为1的维度
z = torch.tensor([1, 2, 3])
print(z.shape)
print(z.unsqueeze(0).shape)  #在0号位置加一维
print(z.unsqueeze(1).shape)  #在1号位置加一维
print("---------------------------")

#删除维度:squeeze去掉所有大小为1的维度
w = torch.ones(1, 3, 1)
print(w.shape)
print(w.squeeze().shape)
print(w.squeeze(0).shape)  #只去掉0号位置
