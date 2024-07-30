import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        # Calculate the size of the encoder output
        self.encoder_output_size = 64 * 7 * 7

        # Dense bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder_output_size, 128),
            nn.ReLU(True),
            nn.Linear(128, self.encoder_output_size),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = x.view(-1, 64, 7, 7)
        x = self.decoder(x)
        return x.view(original_shape)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)


def run_autoencoder(model, epochs):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    mnist_dataset_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 140
    train_loader = torch.utils.data.DataLoader(mnist_dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=5, shuffle=False)

    train_size = len(mnist_dataset_train)
    test_size = len(mnist_dataset_test)
    print(f'Train dataset size: {train_size}')
    print(f'Test dataset size: {test_size}')

    def add_noise(img, noise_factor=0.2):
        noise = noise_factor * torch.randn(*img.shape).to(device)
        noisy_img = img + noise
        noisy_img = torch.clip(noisy_img, 0., 1.)
        return noisy_img

    autoencoder = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(autoencoder.parameters(), lr=0.001)

    for epoch in range(epochs):
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
            noisy_img = add_noise(img)

            output = autoencoder(noisy_img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    autoencoder.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
            noisy_img = add_noise(img)
            output = autoencoder(noisy_img)
            test_loss += criterion(output, img).item()

    test_loss /= len(test_loader)
    print(f'Test set Mean Absolute Error: {test_loss:.4f}')

    indices = np.random.choice(len(mnist_dataset_test), 5, replace=False)
    selected_images = [mnist_dataset_test[i][0] for i in indices]

    with torch.no_grad():
        selected_images = [img.view(-1).unsqueeze(0).to(device) for img in selected_images]
        noisy_selected_images = [add_noise(img) for img in selected_images]
        denoised_images = [autoencoder(img).cpu() for img in noisy_selected_images]

    def imshow(img, ax, title):
        img = img.view(28, 28).cpu().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(f"Conv + Dense / Test Loss: {test_loss}", fontsize=26)

    for i in range(5):
        imshow(selected_images[i].cpu(), axes[0, i], 'Original Image')
        imshow(noisy_selected_images[i].cpu(), axes[1, i], 'Noisy Image')
        imshow(denoised_images[i], axes[2, i], 'Denoised Image')

    plt.tight_layout()
    plt.show()


run_autoencoder(DAE(), 5)
