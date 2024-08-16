# Adapted from transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define hyperparameters
batch_size = 128
image_size = 28
patch_size = 4
num_patches = (image_size // patch_size) ** 2
n_embd = 128
n_head = 8
n_layer = 4
dropout = 0.1
num_classes = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)


class Dense(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        K = self.key(x)
        Q = self.query(x)

        attn = (Q @ K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        V = self.value(x)
        out = attn @ V
        return out, attn


class MultiAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out_list = []
        attn_list = []
        for head in self.heads:
            out, attn = head(x)
            out_list.append(out)
            attn_list.append(attn)
        out = torch.cat(out_list, dim=-1)
        out = self.dropout(self.projection(out))
        return out, attn_list


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


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, n_embd, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, n_embd))
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        attn_weights = []
        for block in self.blocks:
            x, weights = block(x)
            attn_weights.append(weights)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x, attn_weights


def train(model, train_loader, optimizer, scheduler, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    scheduler.step()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f' Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')


def visualize_attention(image, attention_weights, pred_class):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(image.squeeze(), cmap='gray')
    axs[0, 0].set_title(f'Original Image\nPredicted: {pred_class}')
    axs[0, 0].axis('off')

    for i, layer_weights in enumerate(attention_weights[:5]):  # Visualize first 5 layers
        weights = layer_weights[0][0, 0, 1:].reshape(7, 7).detach().cpu().numpy()
        ax = axs[(i + 1) // 3, (i + 1) % 3]
        im = ax.imshow(weights, cmap='viridis')
        ax.set_title(f'Attention Layer {i + 1}')
        ax.axis('off')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ViT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

num_epochs = 20
for epoch in range(1, num_epochs + 1):
    train(model, train_loader, optimizer, scheduler, epoch)
    test(model, test_loader)

torch.save(model.state_dict(), 'model_params_MNIST_2.pth')


# Visualize attention maps.
model.eval()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
num_examples = 5

with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        if i >= num_examples:
            break
        data = data.to(device)
        output, attn_weights = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        visualize_attention(data[0].cpu(), attn_weights, pred.item())