"""
day3.1.2.py —— 从零实现一个字符级小型 GPT（基于《出师表》）

训练语料：诸葛亮《出师表》全文（含标点，见 chushi_biao.txt）
模型结构：Token Embedding + 可学习位置编码 + 4 层 Transformer Block（因果多头自注意力 + MLP + LayerNorm + 残差）
训练方式：AdamW + CrossEntropyLoss，batch_size=32，训练 3000 步，每 500 步打印训练/验证 loss
文本生成：给定起始字符串（如"臣亮言"），用 softmax + torch.multinomial 逐字符采样续写 200 字
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ==================== 0. 设备检测：优先使用 GPU 加速 ====================

# 有 NVIDIA GPU 且安装了 CUDA 版 PyTorch 时，自动用 GPU 训练；否则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 1. 数据准备：字符级分词 ====================

def load_corpus(path, repeat=10):
    """
    读取《出师表》文本，保留标点，按字符粒度分词：每个汉字或标点就是 1 个 token。
    《出师表》全文只有约 800 字，单篇数据量太小，故重复若干遍以增加训练数据量。
    """
    with open(path, "r", encoding="utf-8") as f:
        corpus = f.read().strip()
    return corpus * repeat

corpus = load_corpus("chushi_biao.txt")
print(f"语料总长度: {len(corpus)} 个字符（原文约 {len(corpus) // 10} 字，重复 10 次）")

# 收集语料中出现的所有字符，构建字符表
chars = sorted(set(corpus))
vocab_size = len(chars)
print(f"字符表大小: {vocab_size}, 字符: {''.join(chars)}")

# 字符 <-> 索引 的双向映射
stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 索引（查表用）
itos = {i: ch for i, ch in enumerate(chars)}   # 索引 -> 字符

# 全文转为 torch.long 张量（直接创建在目标设备上），并按 9:1 划分训练集与验证集
data = torch.tensor([stoi[c] for c in corpus], dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"训练集: {len(train_data)} 字符, 验证集: {len(val_data)} 字符")

block_size = 64   # 上下文长度：模型每次只看最近 64 个字符

# ==================== 2. 模型组件 ====================

class CausalSelfAttention(nn.Module):
    """因果多头自注意力：每个位置只能看到它自己及之前的 token（用下三角掩码实现）"""

    def __init__(self, embed_size, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = embed_size // n_heads
        self.qkv = nn.Linear(embed_size, 3 * embed_size)   # 一次线性变换同时算出 Q/K/V
        self.proj = nn.Linear(embed_size, embed_size)      # 多头输出投影回 embed_size
        # 因果掩码：下三角矩阵，右上角全 0（表示未来位置不可见）
        # 用 register_buffer 注册，随模型一起保存/迁移设备，但不是可学习参数
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape   # B: batch, T: 序列长度, C: 嵌入维度
        # 一次算 Q/K/V，再按最后一维切成三份
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # 各为 [B, T, C]
        # 拆成多头：[B, T, n_heads, head_size] -> [B, n_heads, T, head_size]
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        # 缩放点积注意力：Q @ K^T / sqrt(head_size)，数值更稳定
        attn = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)   # [B, n_heads, T, T]
        # 用下三角掩码把未来位置掩成 -inf，softmax 后概率为 0
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)                        # 注意力权重
        out = attn @ v                                            # [B, n_heads, T, head_size]
        # 合并多头回 [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class MLP(nn.Module):
    """前馈网络：对每个位置独立做两层全连接 + GELU 激活（内部扩到 4 倍再缩回）"""

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
    """Transformer 块：注意力 + 前馈，各带 LayerNorm 和残差连接（Pre-Norm 结构）"""

    def __init__(self, embed_size, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)                        # 归一化（用于注意力前）
        self.attn = CausalSelfAttention(embed_size, n_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_size)                        # 归一化（用于前馈前）
        self.mlp = MLP(embed_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 残差连接：x + 注意力输出
        x = x + self.mlp(self.ln2(x))    # 残差连接：x + 前馈输出
        return x

class MiniGPT(nn.Module):
    """小型 GPT：token 嵌入 + 可学习位置编码 + N 个 Transformer 块 + 输出层"""

    def __init__(self, vocab_size, embed_size=128, n_heads=4, n_layers=4, block_size=64):
        super().__init__()
        self.block_size = block_size
        # 可学习嵌入：每个字符索引 -> 向量；每个位置索引 -> 向量
        self.token_embedding = nn.Embedding(vocab_size, embed_size)     # [vocab_size, embed]
        self.position_embedding = nn.Embedding(block_size, embed_size)  # [block_size, embed]
        self.blocks = nn.Sequential(*[Block(embed_size, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_size)                            # 输出前的最后归一化
        self.head = nn.Linear(embed_size, vocab_size)                   # 输出每个字符的 logits

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                  # [B, T, embed] 字符嵌入
        pos = torch.arange(T, device=idx.device)             # 位置 0..T-1
        pos_emb = self.position_embedding(pos)               # [T, embed] 位置嵌入
        x = tok_emb + pos_emb                                # 两者相加作为 Transformer 输入
        x = self.blocks(x)
        return self.head(self.ln_f(x))                       # [B, T, vocab_size]

# ==================== 3. 训练 ====================

def get_batch(data, batch_size=32):
    """随机取一批长度为 block_size 的序列；输入和目标各错开一个字符（下一字符预测）"""
    # 随机索引也要生成在 data 所在的设备上，否则 GPU 张量无法被 CPU 索引
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i + block_size] for i in ix])            # 输入：前 64 个字符
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])    # 目标：后移 1 位的 64 个字符
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, loss_fn, batch_size=32):
    """在训练集和验证集上各评估 20 批的平均损失，用于观察是否过拟合"""
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(20):
            x, y = get_batch(data, batch_size)
            logits = model(x)
            total += loss_fn(logits.view(-1, vocab_size), y.view(-1)).item()
        losses[name] = total / 20
    model.train()
    return losses

torch.manual_seed(42)            # 固定随机种子，保证结果可复现
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)   # 同时固定 CUDA 端的随机种子
model = MiniGPT(vocab_size).to(device)   # 把模型参数迁移到 GPU 上（无 GPU 则留在 CPU）
print(f"MiniGPT 参数量: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=1e-3)      # AdamW 优化器
loss_fn = nn.CrossEntropyLoss()                           # 交叉熵损失

print("开始训练...")
for step in range(3000):
    x, y = get_batch(train_data)
    logits = model(x)                                      # 前向传播 [B, T, vocab_size]
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))  # 展平后计算交叉熵
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 500 == 0:
        losses = estimate_loss(model, train_data, val_data, loss_fn)
        print(f"step {step + 1:4d}, train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

# 保存模型权重
torch.save(model.state_dict(), "model_weights.pth")
print("模型权重已保存到 model_weights.pth")

# ==================== 4. 文本生成 ====================

def generate(model, start="臣亮言", max_new_tokens=200, temperature=1.0):
    """
    给定起始字符串，逐字符续写。
    - 取最近 block_size 个字符作为上下文（负索引切片）
    - 用 softmax 把 logits 转成概率，再用 torch.multinomial 按概率采样下一个字符
    - temperature 越低，输出越保守（可调成 0.8 让古文更连贯）
    """
    model.eval()
    idx = torch.tensor([[stoi[c] for c in start]], dtype=torch.long, device=device)  # 起始字符串转为索引（在目标设备上生成）
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size:]                     # 只取最近 block_size 个字符
            logits = model(idx_cond)[:, -1, :] / temperature          # 只取最后一个位置的预测
            probs = torch.softmax(logits, dim=-1)                     # logits -> 概率分布
            next_token = torch.multinomial(probs, num_samples=1)      # 按概率采样一个字符
            idx = torch.cat([idx, next_token], dim=1)                 # 拼接到序列末尾
    return "".join(itos[i.item()] for i in idx[0])

# 生成演示：以《出师表》开篇之语"臣亮言"为起点，续写 200 个字符
print("\n" + "=" * 60)
print("生成文本演示（起始：臣亮言）:")
print("=" * 60)
print(generate(model))
