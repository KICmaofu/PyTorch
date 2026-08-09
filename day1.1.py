import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import Counter, OrderedDict
from datasets import load_dataset
from torch.utils.data import DataLoader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 IMDB 数据集（HuggingFace datasets，替代已废弃的 torchtext）
print("正在加载 IMDB 数据集...")
ds = load_dataset("stanfordnlp/imdb")
train_data = ds['train']
test_data = ds['test']

# 简单英文分词器（等价于 torchtext 的 basic_english）
def tokenizer(text):
    text = text.lower()
    return re.findall(r"\w+", text)

# 构建词表
def yield_tokens(data):
    for example in data:
        yield tokenizer(example['text'])

print("正在构建词表...")
counter = Counter()
for tokens in yield_tokens(train_data):
    counter.update(tokens)
# 按频率降序排列，保留前 25000 个词
most_common = counter.most_common(25000 - 2)
vocab = OrderedDict()
vocab['<unk>'] = 0
vocab['<pad>'] = 1
for idx, (word, _) in enumerate(most_common, start=2):
    vocab[word] = idx

unk_idx = vocab['<unk>']
pad_idx = vocab['<pad>']

def text_to_ids(text):
    tokens = tokenizer(text)
    return [vocab.get(t, unk_idx) for t in tokens]

# 文本转数值序列，并等长填充
def collate_batch(batch):
    label_list, text_list, length_list = [], [], []
    for example in batch:
        label_list.append(example['label'])
        ids = text_to_ids(example['text'])
        length_list.append(len(ids))
        text_list.append(torch.tensor(ids, dtype=torch.long))

    max_len = max(length_list)
    padded = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
    return torch.tensor(label_list, dtype=torch.long), padded, torch.tensor(length_list)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_batch)

# 定义 LSTM 分类器
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)                     # [batch, seq_len, embed_size]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)          # hidden: [num_layers, batch, hidden_size]
        out = self.fc(hidden[-1])                   # 取最后一层的最后时间步
        return out

model = LSTMModel(len(vocab), embed_size=100, hidden_size=128, num_layers=2, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练 2 个 epoch
print(f"使用设备: {device}，开始训练...")
for epoch in range(2):
    total_loss = 0
    for labels, texts, lengths in train_loader:
        labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for labels, texts, lengths in test_loader:
        labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
        outputs = model(texts, lengths)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"测试准确率: {100 * correct / total:.2f}%")
