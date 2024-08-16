import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter

torch.manual_seed(1975)

device = "cpu"


class Net(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super(Net, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.fc1 = nn.Linear(input_sz, hidden_sz)
        self.fc2 = nn.Linear(hidden_sz, 10)

    def forward(self, x):
        x = x.view(-1, self.input_sz )
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
target_counts = Counter(train_dataset.targets)


def train(i_size, h_size, savename):
    model = Net(i_size, h_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    weight_changes = torch.zeros(10, model.hidden_sz, model.input_sz).to(device)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            old_weights = model.fc1.weight.data.clone()

            loss.backward()
            optimizer.step()

            for class_c in range(10):
                if class_c in target:
                    squared_diff = (model.fc1.weight.data - old_weights) ** 2
                    weight_changes[class_c] += squared_diff

            if batch_idx % 100 == 0 or batch_idx > 59999:
                print(batch_idx)

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    print(weight_changes.shape)
    torch.save(weight_changes.cpu(), f'{savename}.pt')

    plot_heatmaps(savename, h_size)


def plot_one(weight_changes, min_change, max_change, savename):
    class_names = [
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
        'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
    ]

    classes_to_plot = list(range(10))
    for i, t_index in enumerate(classes_to_plot):
        plt.imshow(weight_changes[t_index].cpu().numpy(), cmap='inferno', aspect=1,
                   interpolation='none', vmin=min_change, vmax=max_change)
        plt.title(class_names[t_index])
        plt.savefig(f"{savename}_{t_index}.png", dpi=400)
        plt.clf()
        # plt.show()


def plot_heatmaps(savename, size, percentiles=((0.01, 0.99), (0.1, 0.9))):
    weight_changes = torch.load(f'{savename}.pt')

    for label, count in target_counts.items():
        weight_changes[label] /= count

    min_change = torch.min(weight_changes).item()
    max_change = torch.max(weight_changes).item()
    print(min_change, max_change)

    plot_one(weight_changes, min_change, max_change, f"figs/{savename}_{size}_noNorm")

    for p in percentiles:
        weight_changes_flat = weight_changes.view(-1)
        min_change = torch.quantile(weight_changes_flat, p[0]).item()
        max_change = torch.quantile(weight_changes_flat, p[1]).item()
        print(min_change, max_change)

        plot_one(weight_changes, min_change, max_change, f"figs/{savename}_{size}_{p[1]}")


# train(i_size=32 * 32, h_size=32 * 14, savename='cfair10')
plot_heatmaps('cfair10', 32*14)
