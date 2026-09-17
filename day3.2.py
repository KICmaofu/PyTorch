#验证广播机制:从右往左逐维对齐,相等或一方为1才可广播,结果各维取较大值
import torch

#1.基本规则:(2,3)加(3,),末尾维3匹配,b被广播到每一行
a = torch.arange(6.).reshape(2, 3)
b = torch.tensor([10., 20., 30.])
print(a.shape, b.shape)  #torch.Size([2, 3]) torch.Size([3])
print(a + b)
print("---------------------------")

#2.验证机制:广播等效于把b先复制成(2,3)再逐元素算,三种写法结果一致
b_ex = b.unsqueeze(0).expand(2, 3)      #显式扩展(视图,不拷数据)
b_rp = b.repeat(2, 1)                   #真实复制
print(torch.allclose(a + b_ex, a + b))  #True
print(torch.allclose(a + b_rp, a + b))  #True
print("---------------------------")

#3.两维同时扩展:(3,1)乘(1,4),列向量和行向量互相补全成(3,4)
c = torch.tensor([1., 2., 3.]).reshape(3, 1)          #列
d = torch.tensor([10., 20., 30., 40.]).reshape(1, 4)  #行
e = c * d
print(e.shape)  #torch.Size([3, 4])
print(e)        #第i行来自第i个c,第j列来自第j个d
manual = torch.tensor([[c[i, 0].item() * d[0, j].item() for j in range(4)] for i in range(3)])
print(torch.allclose(e, manual))  #True 逐元素手算一致
print("---------------------------")

#4.维度数不同:先右对齐,缺的维补1再逐维比较
f = torch.ones(2, 1, 3)
g = torch.arange(3.)                #(3,)按(1,1,3)对齐
print((f + g).shape)                #torch.Size([2, 1, 3])
h = torch.arange(2.).reshape(2, 1)  #(2,1)按(1,2,1)对齐
print((f + h).shape)                #torch.Size([2, 2, 3]) dim1的1和2互相扩展
print("---------------------------")

#5.数值验证:手写期望结果,与广播结果对照
x = torch.arange(6.).reshape(2, 3)
y = torch.tensor([100., 200., 300.])
expect = torch.tensor([[100., 201., 302.], [103., 204., 305.]])
print(torch.allclose(x + y, expect))  #True 每行=x行+y
print("---------------------------")

#6.expand vs repeat:数值一致,但expand是stride=0的视图,不占新内存
p = torch.tensor([[1.], [2.]])  #(2,1)
p_ex = p.expand(2, 3)
p_rp = p.repeat(1, 3)
print(p_ex)
print(torch.allclose(p_ex, p_rp))  #True 数值一样
print(p_ex.stride())  #(1, 0) stride含0=同一元素被重复读,这就是零拷贝的实现
print(p_ex.untyped_storage().data_ptr() == p.untyped_storage().data_ptr())  #True 共享存储
print(p_rp.untyped_storage().data_ptr() == p.untyped_storage().data_ptr())  #False 独立存储
print("---------------------------")

#7.反面案例:3和2都不是1,无法广播,直接报错
m1 = torch.ones(2, 3)
m2 = torch.ones(2, 2)
# print(m1 + m2)  #报错:The size of tensor a (3) must match the size of tensor b (2)
print("---------------------------")

#8.应用场景:权重逐通道缩放图片,(C,1,1)广播到(N,C,H,W)
img = torch.ones(2, 3, 4, 4)  #N=2, C=3, H=W=4
w = torch.tensor([1., 2., 3.]).reshape(3, 1, 1)
out = img * w
print(out.shape)        #torch.Size([2, 3, 4, 4])
print(out[0, 1, 0, 0])  #tensor(2.) 通道1乘2
print(out[1, 2, 3, 3])  #tensor(3.) 通道2乘3
print("---------------------------")

#9.总结:广播=从右往左逐维比较(相等或一方为1),为1或缺维则虚拟扩展,结果形状各维取大,底层stride=0零拷贝
