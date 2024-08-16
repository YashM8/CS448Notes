import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


class Autoencoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.bottleneck_size = bottleneck_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * input_shape[1] * input_shape[2], 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, bottleneck_size)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64 * input_shape[1] * input_shape[2]),
            nn.GELU(),
            nn.Unflatten(1, (64, input_shape[1], input_shape[2])),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


input_shape = (3, 256, 256)
bottleneck_size = 75

model = Autoencoder(input_shape, bottleneck_size)
print(model)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"DEVICE: {device}")
model.to(device)

# Load data
data = []
for i in range(300):
    img_path = f"/Users/ypm/Desktop/drive-download-20240801T184124Z-001/Code/denoise/profiles/color_{i+1}.png"
    img = Image.open(img_path).resize((256, 256))
    img = np.array(img)
    data.append(img)
print(len(data))
np.random.seed(1975)
np.random.shuffle(data)

# Split data into train and test sets
train_data = data[:90]
test_data = data[90:]

# Add noise to the data
train_data_noisy = [img.copy() for img in train_data]
test_data_noisy = [img.copy() for img in test_data]

for i in range(len(train_data_noisy)):
    noise = np.round(np.random.normal(0, 6, size=train_data_noisy[i].shape)).astype(np.int64)
    noisy_img = train_data_noisy[i] + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    train_data_noisy[i] = noisy_img

for i in range(len(test_data_noisy)):
    noise = np.round(np.random.normal(0, 6, size=test_data_noisy[i].shape)).astype(np.int64)
    noisy_img = test_data_noisy[i] + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    test_data_noisy[i] = noisy_img

# Convert data to tensors
train_data_tensor = torch.from_numpy(np.array(train_data_noisy)).float() / 255
test_data_tensor = torch.from_numpy(np.array(test_data_noisy)).float() / 255
train_target_tensor = torch.from_numpy(np.array(train_data)).float() / 255
test_target_tensor = torch.from_numpy(np.array(test_data)).float() / 255

train_data_tensor = train_data_tensor.permute(0, 3, 1, 2)
test_data_tensor = test_data_tensor.permute(0, 3, 1, 2)
train_target_tensor = train_target_tensor.permute(0, 3, 1, 2)
test_target_tensor = test_target_tensor.permute(0, 3, 1, 2)

train_data_tensor = train_data_tensor.to(device)
train_target_tensor = train_target_tensor.to(device)
test_data_tensor = test_data_tensor.to(device)
test_target_tensor = test_target_tensor.to(device)

epochs = 500


def train():
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    num_epochs = epochs
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(train_data_tensor)
        loss = criterion(outputs, train_target_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    with torch.no_grad():
        train_output = model(train_data_tensor)
        test_output = model(test_data_tensor)

    train_loss = criterion(train_output, train_target_tensor)
    test_loss = criterion(test_output, test_target_tensor)

    print(f'Train Reconstruction Error: {train_loss.item():.4f}')
    print(f'Test Reconstruction Error: {test_loss.item():.4f}')

    torch.save(model.state_dict(), f'bottleneck_{bottleneck_size}_{epochs}_NEW.pth')


train()

# model.load_state_dict(torch.load(f'bottleneck_{bottleneck_size}_{epochs}.pth'))
# model.eval()
#
# print(f'bottleneck_{bottleneck_size}_{epochs}.pth')
# # Prepare data for visualization
# index = 1
# test_example = test_data_tensor[index].unsqueeze(0).to(device)
# test_target = train_target_tensor[index].unsqueeze(0).to(device)
#
# with torch.no_grad():
#     reconstructed_example = model(test_example)
#
# # Convert tensors to numpy arrays
# test_example_np = test_example.cpu().squeeze().permute(1, 2, 0).numpy()
# reconstructed_example_np = reconstructed_example.cpu().squeeze().permute(1, 2, 0).numpy()
# test_target_np = test_target.cpu().squeeze().permute(1, 2, 0).numpy()
#
# # Normalize pixel values to [0, 1] for visualization
# test_example_np = (test_example_np - test_example_np.min()) / (test_example_np.max() - test_example_np.min())
# reconstructed_example_np = (reconstructed_example_np - reconstructed_example_np.min()) / (reconstructed_example_np.max() - reconstructed_example_np.min())
# test_target_np = (test_target_np - test_target_np.min()) / (test_target_np.max() - test_target_np.min())
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
#
# ax1.imshow(test_example_np)
# ax1.set_title("Input w Noise (6)", fontsize=24)
# ax1.axis('off')
#
# ax2.imshow(reconstructed_example_np)
# ax2.set_title("Denoised", fontsize=24)
# ax2.axis('off')
#
# ax3.imshow(test_target_np)
# ax3.set_title("Target", fontsize=24)
# ax3.axis('off')
#
# plt.show()
