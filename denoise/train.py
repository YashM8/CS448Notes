import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from dataset import CustomImageDataset
from denoiser import DenoisingAutoencoderLeaky

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

model = DenoisingAutoencoderLeaky().to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0006)

# checkpoint_path = 'model_denoise_NP_leaky.pth'
# model.load_state_dict(torch.load(checkpoint_path))

NOISE_LEVEL = 0.06

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = CustomImageDataset('test_blue', transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=28, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=28, shuffle=False)


def train(num_epochs=400, patience=15):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            noisy_inputs = inputs + NOISE_LEVEL * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader)}')

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                noisy_inputs = inputs + NOISE_LEVEL * torch.randn_like(inputs)
                noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

                outputs = model(noisy_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f'Validation Loss: {val_loss}')

        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'model_denoise_NP_leaky_moreNoise.pth')
        else:
            epochs_without_improvement += 1

        # Check if early stopping criteria is met
        if epochs_without_improvement >= patience:
            print('Early stopping triggered.')
            break


# train()


model.to("cpu")
model.load_state_dict(torch.load('model_denoise_NP_leaky.pth'))
model.eval()


def show_images(original, noisy, denoised):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Image')
    axes[1].imshow(noisy.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Noisy Image')
    axes[2].imshow(denoised.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title('Denoised Image')
    for ax in axes:
        ax.axis('off')
    plt.show()


# Get a batch of test data
dataiter = iter(test_loader)
images, _ = next(dataiter)
images = images.to("cpu")

# Add noise to the images
noisy_images = images + NOISE_LEVEL * torch.randn_like(images)
noisy_images = torch.clamp(noisy_images, 0., 1.)

# Get denoised images
with torch.no_grad():
    denoised_images = model(noisy_images)

# Show images
for i in range(5):  # Show 5 examples
    show_images(images[i], noisy_images[i], denoised_images[i])


def evaluate(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            noisy_inputs = inputs + NOISE_LEVEL * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)
    return average_test_loss


# Evaluate the model
test_loss = evaluate(test_loader, model, criterion, "cpu")
print(f'Test Loss: {test_loss:.4f}')
