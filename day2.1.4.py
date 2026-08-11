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

# ============ 小型 GPT：从零实现一个字符级语言模型 ============
# 前面学的索引知识在这里都会用到：
#  - 因果掩码 = torch.tril 生成的下三角矩阵（右上角被 mask 掉）
#  - 生成时 idx[:, -block_size:] 用负索引取最近的上下文

import torch.nn as nn
import torch.optim as optim

# ---------- 1. 数据准备：字符级分词 ----------
# 字符级模型不需要分词器：每个字符（含空格/标点）就是一个 token
corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the lazy dog sleeps under the sun. "
    "the sun rises over the quiet forest. "
    "a fox jumps and runs through the trees. "
) * 20   # 重复 20 次，凑足训练数据

chars = sorted(set(corpus))
vocab_size = len(chars)
print(f"\n字符表大小: {vocab_size}, 字符: {''.join(chars)}")

stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 索引（查表）
itos = {i: ch for ch, i in stoi.items()}       # 索引 -> 字符

data = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 64   # 上下文长度：模型每次只看最近 64 个字符

# ---------- 2. 模型组件 ----------

class CausalSelfAttention(nn.Module):
    """因果多头自注意力：每个位置只能看到它自己及之前的 token"""
    def __init__(self, embed_size, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = embed_size // n_heads
        self.qkv = nn.Linear(embed_size, 3 * embed_size)   # 一次算出 Q/K/V
        self.proj = nn.Linear(embed_size, embed_size)
        # 因果掩码：下三角矩阵，右上角全 0（表示未来位置不可见）
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # [B, T, C] 各一份
        # 拆成多头：[B, T, n_heads, head_size] -> [B, n_heads, T, head_size]
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        # 缩放点积注意力：Q @ K^T / sqrt(head_size)
        attn = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)   # [B, n_heads, T, T]
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # 掩掉未来位置
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v                                    # [B, n_heads, T, head_size]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 合并多头
        return self.proj(out)

class MLP(nn.Module):
    """前馈网络：逐位置的两层全连接 + GELU 激活"""
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer 块：注意力 + 前馈，各带残差连接和层归一化"""
    def __init__(self, embed_size, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention(embed_size, n_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 残差连接：x + 注意力输出
        x = x + self.mlp(self.ln2(x))    # 残差连接：x + 前馈输出
        return x

class MiniGPT(nn.Module):
    """小型 GPT：token 嵌入 + 位置嵌入 + N 个 Transformer 块 + 输出层"""
    def __init__(self, vocab_size, embed_size=64, n_heads=4, n_layers=2, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)    # token -> 向量
        self.position_embedding = nn.Embedding(block_size, embed_size) # 位置 -> 向量
        self.blocks = nn.Sequential(*[Block(embed_size, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)   # 输出每个字符的概率

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                          # [B, T, embed]
        pos = torch.arange(T, device=idx.device)                     # 位置 0..T-1
        pos_emb = self.position_embedding(pos)                       # [T, embed]
        x = tok_emb + pos_emb                                        # 词向量 + 位置向量
        x = self.blocks(x)
        return self.head(self.ln_f(x))    # [B, T, vocab_size]

# ---------- 3. 训练 ----------
def get_batch(data, batch_size=32):
    """随机取一批长度为 block_size 的序列；输入和目标各错开一个字符"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

torch.manual_seed(42)
model = MiniGPT(vocab_size)
print(f"MiniGPT 参数量: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

print("开始训练...")
for step in range(3000):
    x, y = get_batch(train_data)
    logits = model(x)                                   # [B, T, vocab_size]
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 500 == 0:
        print(f"step {step + 1:4d}, loss = {loss.item():.4f}")

# ---------- 4. 文本生成 ----------
def generate(model, start="the ", max_new_tokens=100):
    """给定开头，逐个字符预测下一个字符（多项式采样）"""
    model.eval()
    idx = torch.tensor([[stoi[c] for c in start]], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size:]   # 负索引：只取最近 block_size 个字符
            logits = model(idx_cond)[:, -1, :]      # 只取最后一个位置的预测
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # 按概率采样
            idx = torch.cat([idx, next_token], dim=1)              # 拼接到序列末尾
    return "".join(itos[i.item()] for i in idx[0])

print("\n生成的文本:")
print(generate(model))