# Data and explanations from Andreaj Karpathy's Lectures.

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 14 * 14 * 2
block_size = 32
iters = 2500
lr = 1e-3

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

eval_iters = 100
n_embd = 64
n_head = 4
n_layer = 3
dropout = 0.2


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
encode_dict = {ch: i for i, ch in enumerate(chars)}
decode_dict = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [encode_dict[c] for c in s]


def decode(l):
    return ''.join([decode_dict[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Dense(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)  # (B, T, size)
        Q = self.query(x)  # (B, T, size)

        # (B, T, size) @ (B, size, T) -> (B, T, T)
        scores = Q @ K.transpose(-2, -1) * K.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # (B, T, T)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # (B, T, size)
        V = self.value(x)

        # (B, T, T) @ (B, T, size) -> (B, T, size)
        out = probs @ V

        return out


class MultiAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head
        self.self_attention = MultiAttention(n_head, head_size)
        self.dense = Dense(n_embd)
        self.norm_attention = nn.LayerNorm(n_embd)
        self.norm_linear = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.norm_attention(x))
        x = x + self.dense(self.norm_linear(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_table = nn.Embedding(vocab_size, n_embd)
        self.position_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.final = nn.Linear(n_embd, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.025)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # (B, T, C)
        tok_emb = self.token_table(idx)
        pos_emb = self.position_table(torch.arange(T, device=device))

        # (B, T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)

        # (B, T, vocab_size)
        logits = self.final(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            # (B, C)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


model = Transformer()
m = model.to(device)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)


m.apply(init_weights)

print(f"Number of parameters - \n {sum(p.numel() for p in m.parameters()) / 1e6} M parameters \n")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iteration in range(iters):

    if iteration % 10 == 0 or iteration == iters:
        losses = estimate_loss()
        print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(m.state_dict(), 'model_params_2.pth')

# generate.
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
