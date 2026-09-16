#4维张量[batch, seq, head, dim]的维度调换与精准切片
import torch

#创建4维张量:batch=2, seq=3, head=4, dim=5(用arange填充,方便观察元素)
x = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
print(x.shape)  #torch.Size([2, 3, 4, 5])
print("---------------------------")

#1.transpose:只交换两个维度,注意力常用[batch, seq, head, dim]->[batch, head, seq, dim]
x_t = x.transpose(1, 2)
print(x_t.shape)  #torch.Size([2, 4, 3, 5])
print("---------------------------")

#transpose是视图:共享存储,改一个另一个跟着变
print(x.data_ptr() == x_t.data_ptr())  #True:转置不产生偏移,data_ptr相同
x_t[0, 0, 0, 0] = 999
print(x[0, 0, 0, 0])  #tensor(999) 原张量被改,证明共享
print("---------------------------")

#2.permute:一次性重排多个维度,参数是新顺序各位置对应原张量的维度号
x_p = x.permute(1, 0, 2, 3)  #第0维取原第1维,第1维取原第0维,后两维不动
print(x_p.shape)  #torch.Size([3, 2, 4, 5])
print("---------------------------")

#3.转置后不连续:直接view会报错,需先contiguous拷贝成连续
y = torch.arange(24).reshape(2, 3, 4)
y_t = y.transpose(0, 1)
print(y_t.is_contiguous())  #False
# print(y_t.view(24))  #报错:view size is not compatible with input tensor's size and stride
print(y_t.contiguous().view(24).shape)  #torch.Size([24]) 先拷贝成连续再变形
print("---------------------------")

#4.精准切片:整数索引固定某维会消掉该维,切片写法则保留该维
z = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
print(z[0].shape)       #torch.Size([3, 4, 5]) 取第0个batch,降维
print(z[0:1].shape)     #torch.Size([1, 3, 4, 5]) 切片保留batch维
print(z[:, 0].shape)    #torch.Size([2, 4, 5]) 只取第0个token
print(z[:, 0:1].shape)  #torch.Size([2, 1, 4, 5]) 切片保留seq维
print("---------------------------")

#5.多维度组合切片:第0个batch+所有seq+第1个head+前3个dim
s = z[0, :, 1, :3]
print(s.shape)  #torch.Size([3, 3])
print(s)
print("---------------------------")

#6....省略号占位,自动展开中间的所有维度,与上面等价
s2 = z[0, ..., 1, :3]
print(s2.shape)  #torch.Size([3, 3])
print("---------------------------")

#7.步长切片:seq维度隔一个取一个
s3 = z[:, ::2]
print(s3.shape)  #torch.Size([2, 2, 4, 5]) seq=3只取索引0,2
print("---------------------------")

#8.切片是视图:有偏移时data_ptr不同,比较底层存储才准确
sl = z[1]  #取第1个batch,存储偏移=60
print(z.data_ptr() == sl.data_ptr())  #False:切片带偏移
print(z.untyped_storage().data_ptr() == sl.untyped_storage().data_ptr())  #True:同一存储
sl[0, 0, 0] = 999
print(z[1, 0, 0, 0])  #tensor(999) 原张量被改,证明共享
print("---------------------------")

#9.综合场景:注意力常见处理流程
q = torch.randn(2, 3, 4, 5)  #batch=2, seq=3, head=4, dim=5
q_t = q.transpose(1, 2)      #->[batch, head, seq, dim],便于每个head独立算注意力
print(q_t.shape)  #torch.Size([2, 4, 3, 5])
first = q_t[:, :, 0, :]  #逐head取第0个token向量
print(first.shape)       #torch.Size([2, 4, 5])
merged = q.view(2, 3, 4 * 5)  #连续张量可直接view合并head与dim
print(merged.shape)           #torch.Size([2, 3, 20])
