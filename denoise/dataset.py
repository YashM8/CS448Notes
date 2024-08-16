from torch.utils.data import Dataset
import os
from PIL import Image


class CustomImageDataset(Dataset):
    # Image size 128 x 128 x 3
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, image


