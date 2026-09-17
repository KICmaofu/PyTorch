#手动实现批量矩阵乘法:[b, n, m] @ [b, m, p] -> [b, n, p]
import torch

#构造数据:batch=2,A每个batch是2x3,B每个batch是3x2
A = torch.tensor([[[1., 2., 3.],
                   [4., 5., 6.]],
                  [[7., 8., 9.],
                   [10., 11., 12.]]])
B = torch.tensor([[[1., 2.],
                   [3., 4.],
                   [5., 6.]],
                  [[7., 8.],
                   [9., 10.],
                   [11., 12.]]])
print(A.shape, B.shape)  #torch.Size([2, 2, 3]) torch.Size([2, 3, 2])
print("---------------------------")

#1.内置参照:torch.bmm,@在三维时等价于bmm,拿它的结果对照手动实现
C_ref = torch.bmm(A, B)
print(C_ref)
print((A @ B).shape, torch.allclose(A @ B, C_ref))  #torch.Size([2, 2, 2]) True
print("---------------------------")

#2.最贴定义:四重循环逐元素算,C[b,i,j] = sum_k A[b,i,k]*B[b,k,j]
b, n, m = A.shape
p = B.shape[-1]
C2 = torch.zeros(b, n, p)
for bi in range(b):
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[bi, i, k].item() * B[bi, k, j].item()  #取标量相乘再累加
            C2[bi, i, j] = s
print(torch.allclose(C2, C_ref))  #True 与内置结果一致
print("---------------------------")

#3.折叠k循环:乘法退化为一维点积,C[b,i,j] = A第i行 · B第j列
C3 = torch.zeros(b, n, p)
for bi in range(b):
    for i in range(n):
        for j in range(p):
            C3[bi, i, j] = torch.dot(A[bi, i], B[bi, :, j])
print(torch.allclose(C3, C_ref))  #True
print("---------------------------")

#4.换视角:矩阵乘法=一组外积之和,只留k循环
#A[:, :, k]是[b,n]的列,B[:, k]是[b,p]的行,外积成[b,n,p]再累加
C4 = torch.zeros(b, n, p)
for k in range(m):
    C4 += A[:, :, k].unsqueeze(2) * B[:, k].unsqueeze(1)
print(torch.allclose(C4, C_ref))  #True
print("---------------------------")

#5.全向量化:广播展开出k轴,相乘后对k轴求和,显式循环全部消失
#A.unsqueeze(3)->[b,n,m,1],B.unsqueeze(1)->[b,1,m,p],乘完是[b,n,m,p]
C5 = (A.unsqueeze(3) * B.unsqueeze(1)).sum(dim=2)
print(C5.shape)                   #torch.Size([2, 2, 2])
print(torch.allclose(C5, C_ref))  #True
print("---------------------------")

#6.einsum对照:下标字符串就是求和公式,bik,bkj->bij表示对k求和且k不出现在输出
print(torch.allclose(torch.einsum("bik,bkj->bij", A, B), C_ref))  #True
print("---------------------------")

#7.应用场景:多头注意力分数Q@K^T,4维批量乘法可摊平成3维bmm
q = torch.randn(2, 4, 3, 5)      #batch=2, head=4, seq=3, dim=5
k = torch.randn(2, 4, 3, 5)
s_ref = q @ k.transpose(-1, -2)  #每个head独立算[3,5]@[5,3]
print(s_ref.shape)               #torch.Size([2, 4, 3, 3])
q_flat = q.reshape(-1, 3, 5)     #前两维合并成8个独立矩阵
k_flat = k.reshape(-1, 3, 5)
s_flat = torch.bmm(q_flat, k_flat.transpose(1, 2)).reshape(2, 4, 3, 3)
print(torch.allclose(s_flat, s_ref))  #True 摊平算完再变回来
s_bc = (q.unsqueeze(3) * k.unsqueeze(2)).sum(-1)  #q补j轴,k补i轴,对dim轴求和
print(torch.allclose(s_bc, s_ref))    #True 广播版与上面同一套路
print("---------------------------")

#8.总结:手动实现本质=对齐内维k->相乘->对k求和,循环/外积/广播/einsum是同一件事的不同写法
