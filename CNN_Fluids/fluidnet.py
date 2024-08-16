import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.init as init


class FluidNet(nn.Module):
    def __init__(self):
        super(FluidNet, self).__init__()

        # Encoder
        self.conv0 = nn.Conv2d(3, 6, kernel_size=2, padding=1)
        self.conv1 = nn.Conv2d(6, 12, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=2, padding=1)

        self.deconv3 = nn.ConvTranspose2d(48, 24, kernel_size=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(24, 12, kernel_size=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(12, 6, kernel_size=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(6, 3, kernel_size=2, padding=1)

        self.deconv7 = nn.ConvTranspose2d(3, 3, kernel_size=1)

        self.relu = nn.LeakyReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x0 = self.relu(self.conv0(x))
        x1 = self.relu(self.conv1(x0))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))

        x = self.relu(self.deconv3(x3))
        x = self.relu(self.deconv4(x))
        x = self.relu(self.deconv5(x))
        x = self.relu(self.deconv6(x))

        x = self.relu(self.deconv7(x))

        return x


# Load the PKL files
dataX_path = 'data/dataX.pkl'
dataY_path = 'data/dataY.pkl'

with open(dataX_path, 'rb') as f:
    dataX = pickle.load(f)

with open(dataY_path, 'rb') as f:
    dataY = pickle.load(f)

dataX = np.array(dataX)
dataY = np.array(dataY)

dataX_train, dataX_test, dataY_train, dataY_test = train_test_split(dataX, dataY, test_size=0.90,
                                                                    random_state=644)
dataX_tensor = torch.tensor(dataX_train, dtype=torch.float32)
dataY_tensor = torch.tensor(dataY_train, dtype=torch.float32)

train_dataset = TensorDataset(dataX_tensor, dataY_tensor)
batch_size = 14*10
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = FluidNet()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print(device)


criterion = nn.MSELoss()
optimizer = torch.optim.NAdam(model.parameters())

## Train the model
# num_epochs = 250
# total_batches = len(train_dataloader)
#
# for epoch in range(num_epochs):
#     print("................................................................")
#     epoch_loss = 0.0
#     batch_count = 0
#
#     for batch_dataX, batch_dataY in train_dataloader:
#         batch_count += 1
#         batch_dataX = batch_dataX.to(device)
#         batch_dataY = batch_dataY.to(device)
#
#         outputs = model(batch_dataX)
#         loss = criterion(outputs, batch_dataY)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#         print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_count}/{total_batches} - Batch Loss: {loss.item():.4f}")
#
#     epoch_loss /= total_batches
#     print(f'Epoch {epoch+1}/{num_epochs} - Average Loss: {epoch_loss:.4f}')

# torch.save(model.state_dict(), 'updated_unet_model_3.pth')

sample_index = 65
channel_index = 0

# plt.imshow(dataX[sample_index, channel_index], cmap='viridis')
# plt.colorbar()
# plt.title(f'dataX - Sample {sample_index}, Channel {channel_index}')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

model.load_state_dict(torch.load('updated_unet_model_2.pth'))
model.eval()

with torch.no_grad():
    example = torch.tensor(dataX_test[sample_index])
    example = example.to(device)
    output = model(example)

plt.figure(figsize=(15, 5))

# Ground truth
plt.subplot(1, 3, 1)
plt.imshow(dataY_test[sample_index, channel_index], cmap='viridis', vmin=0, vmax=0.25)
plt.colorbar()
plt.title(f'Ground Truth - Test Sample {sample_index}')
plt.ylabel('Ux')

# Model output
plt.subplot(1, 3, 2)
plt.imshow(output[channel_index].cpu(), cmap='viridis', vmin=0, vmax=0.25)
plt.colorbar()
plt.title(f'Prediction')
plt.ylabel('Ux')


# Difference
diff = (dataY_train[sample_index, channel_index] - output[0].cpu().numpy())**2
plt.subplot(1, 3, 3)
plt.imshow(diff, cmap='viridis')
plt.colorbar()
plt.title('Squared Error')
plt.ylabel('Ux')


plt.tight_layout()
plt.savefig(f"figures/test_demo_{80}_{0}.jpg")

