import torch
import torch.nn as nn

import torch.nn.init as init

class DenoisingAutoencoderLeaky(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoderLeaky, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 1824)
        self.gelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(1824, 1024)

        self.fc3 = nn.Linear(1024, 1824)
        self.fc4 = nn.Linear(1824, 64 * 16 * 16)
        self.unflatten = nn.Unflatten(1, (64, 16, 16))
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.flatten(enc3)
        enc5 = self.gelu(self.fc1(enc4))
        enc6 = self.fc2(enc5)

        dec1 = self.gelu(self.fc3(enc6))
        dec2 = self.gelu(self.fc4(dec1))
        dec3 = self.unflatten(dec2)
        dec4 = self.decoder1(dec3)
        dec5 = self.decoder2(dec4 + enc2)
        dec6 = self.decoder3(dec5 + enc1)

        return dec6

