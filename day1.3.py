#验证内存共享与拷贝差异
import torch
import numpy as np

#验证工具:data_ptr()返回张量数据的内存地址,地址相同=共享内存
x = torch.tensor([1, 2, 3])
print(x.data_ptr())
print("---------------------------")

#1.from_numpy共享内存:numpy数组地址和张量地址一致
a = np.array([1, 2, 3])
t1 = torch.from_numpy(a)
t2 = torch.tensor(a)
print(t1.data_ptr() == a.ctypes.data)  #True 共享
print(t2.data_ptr() == a.ctypes.data)  #False 复制
print("---------------------------")

#改numpy数据:共享的跟着变,复制的不变
a[0] = 100
print(t1)  #tensor([100, 2, 3])
print(t2)  #tensor([1, 2, 3])
print("---------------------------")

#2.numpy()转换共享内存
c = torch.tensor([1, 2, 3])
n = c.numpy()
print(c.data_ptr() == n.ctypes.data)  #True
c[0] = 100
print(n)  #共享,跟着变
print("---------------------------")

#3.clone()深拷贝:地址不同,互不影响
y = torch.tensor([1, 2, 3])
y_clone = y.clone()
print(y.data_ptr() == y_clone.data_ptr())  #False
y[0] = 100
print(y)        #改了
print(y_clone)  #没变
print("---------------------------")

#4.view()是视图:地址相同,共享内存
z = torch.tensor([1, 2, 3, 4])
z_view = z.view(2, 2)
print(z.data_ptr() == z_view.data_ptr())  #True
z_view[0, 0] = 100
print(z)  #原张量被改
print("---------------------------")

#5.切片也是视图:共享同一块存储(注意data_ptr含偏移量)
s = torch.tensor([1, 2, 3, 4])
s_slice = s[1:3]
print(s.data_ptr() == s_slice.data_ptr())  #False:切片有偏移,data_ptr指向自身首元素
print(s.untyped_storage().data_ptr() == s_slice.untyped_storage().data_ptr())  #True:底层存储相同
s_slice[0] = 100
print(s)  #tensor([1, 100, 3, 4]) 原张量被改,证明共享
print("---------------------------")

#6.detach()共享内存
d = torch.tensor([1., 2.], requires_grad=True)
d_det = d.detach()
print(d.data_ptr() == d_det.data_ptr())  #True
print("---------------------------")

#7.reshape()通常返回视图,但不连续时会拷贝
r = torch.tensor([1, 2, 3, 4])
print(r.data_ptr() == r.reshape(2, 2).data_ptr())  #True 连续,视图
rt = r.view(2, 2).t()  #转置后不连续
print(rt.data_ptr() == rt.reshape(-1).data_ptr())  #False 只能拷贝
print("---------------------------")

#8.总结:想独立就clone,想省内存就用视图/numpy互转
m = torch.arange(5)
print(m.clone().data_ptr() == m.data_ptr())  #False 拷贝
print(m[:].data_ptr() == m.data_ptr())       #True 视图
